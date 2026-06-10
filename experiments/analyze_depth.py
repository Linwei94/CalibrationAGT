"""
Depth / diagnostic analysis for the TPAMI resubmission (TPAMI-2026-03-0977).

Addresses Reviewer 1 (W3, R3) -- "deeper diagnostic analysis of the proposed
calibrators" -- and produces the numbers behind the new theory blocks A2/A3.

It computes four diagnostics for any (dataset, architecture):

  D1  Entropy-bin decomposition
        ECE_true and soft-Brier per annotation-entropy quantile, PER METHOD.
        Shows that the gains of ambiguity-aware calibration are concentrated in
        high-entropy examples (answers "do gains come from high-entropy data?").

  D2  Per-confidence-bin correction
        For each method, the population reliability gap  mean(conf) - mean(pi_pred)
        per confidence bin, before vs. after calibration.  Shows WHICH confidence
        bins each method corrects.

  D3  Class- vs. instance-structured ambiguity
        Per-(voted)class ECE_true, plus eta^2 = fraction of annotation-entropy
        variance explained by the class label.  High eta^2  => ambiguity is
        class-structured => class-aware methods (VS / Dirichlet-Soft) should help;
        low eta^2 => instance-structured => class-aware methods are unnecessary.
        (Answers "are class-aware methods necessary for class-specific ambiguity?")

  D4  LS-TS proxy validity  (pairs with Proposition "LS-TS proxy validity", A3)
        u_i = 1 - pi_{y*}(x_i)  (true disagreement),  v_i = 1 - p_{y*}(x_i) (model
        complement-confidence).  Reports  vbar - ubar  (signed proxy gap, predicts
        over/under-smoothing) and rho = corr(u, v) (predicts residual ECE gap),
        with the fitted SLTS / LS-TS temperatures.

USAGE
-----
  # verify the analysis end-to-end on a synthetic ambiguous dataset (no cache needed):
  python analyze_depth.py --demo

  # real data: first build per-(dataset,arch) bundles with make_bundles.py, then:
  python analyze_depth.py --bundle results/bundles/cifar10h_resnet50.npz
  python analyze_depth.py --bundle-dir results/bundles      # all bundles, writes a summary

A "bundle" is a .npz with arrays:
  logits_cal (Ncal,K) float, logits_te (Nte,K) float,
  soft_cal   (Ncal,K) float, soft_te   (Nte,K) float  (annotator distributions, rows sum to 1),
  hard_cal   (Ncal,)  int   (voted labels),  hard_te (Nte,) int,
and scalar fields  name (str), n_classes (int).   See make_bundles.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    TemperatureScaling, SoftLabelTS, MonteCarloTS, LabelSmoothTS,
    VectorScaling, DirichletCalibration, SoftIsotonicRegression, apply_parametric,
)
from metrics import compute_ece_sampled, annotation_entropy


# ──────────────────────────────────────────────────────────────────────────────
# Method fitting (reuses calibration.py exactly as the main experiments do)
# ──────────────────────────────────────────────────────────────────────────────

def fit_all_methods(b: dict) -> dict:
    """Fit every calibrator on the calibration split and return {name: probs_te}."""
    K = int(b["n_classes"])
    lc = torch.tensor(b["logits_cal"], dtype=torch.float32)
    yh = torch.tensor(b["hard_cal"], dtype=torch.long)
    ys = torch.tensor(b["soft_cal"], dtype=torch.float32)
    lt = b["logits_te"]
    probs_cal_np = torch.softmax(lc, dim=1).numpy()

    out, temps = {}, {}

    out["Uncalibrated"] = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()

    ts = TemperatureScaling().fit(lc, yh);        out["TS"] = apply_parametric(ts, lt);  temps["TS"] = ts.T
    slts = SoftLabelTS().fit(lc, ys);             out["SLTS"] = apply_parametric(slts, lt); temps["SLTS"] = slts.T
    mcts1 = MonteCarloTS(n_samples=1).fit(lc, ys); out["MCTS S=1"] = apply_parametric(mcts1, lt); temps["MCTS S=1"] = mcts1.T
    lsts = LabelSmoothTS().fit(lc, yh);           out["LS-TS"] = apply_parametric(lsts, lt); temps["LS-TS"] = lsts.T
    vs = VectorScaling(K).fit(lc, ys);            out["VS"] = apply_parametric(vs, lt)
    dcs = DirichletCalibration(K).fit_soft(lc, ys); out["Dirichlet-Soft"] = apply_parametric(dcs, lt)

    # IR-Soft is non-parametric: monotone map on top-class confidence -> reconstruct prob matrix
    ir = SoftIsotonicRegression().fit(probs_cal_np, b["soft_cal"])
    probs_te_np = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()
    cal_conf, pred = ir.calibrate(probs_te_np)
    pir = np.zeros_like(probs_te_np)
    for i in range(len(probs_te_np)):
        pir[i, pred[i]] = cal_conf[i]
        rest = np.ones(K) * (1 - cal_conf[i]) / (K - 1)
        rest[pred[i]] = 0
        pir[i] += rest
    out["IR-Soft"] = pir

    return out, temps


# ──────────────────────────────────────────────────────────────────────────────
# D1: entropy-bin decomposition
# ──────────────────────────────────────────────────────────────────────────────

def diag_entropy_bins(probs_by_method: dict, soft_te: np.ndarray,
                      n_q: int = 5, n_bins: int = 15, seed: int = 0) -> dict:
    """ECE_true per annotation-entropy bin, per method.

    Tie-robust: with continuous entropy this is an equal-mass quantile split; when
    entropy is heavily tied (e.g. class-conditional synthetic annotators, where all
    images of a class share one entropy value) it falls back to grouping by distinct
    entropy levels so no bin is spuriously empty.
    """
    H = annotation_entropy(soft_te)
    uniq = np.unique(np.round(H, 6))
    if len(uniq) <= n_q + 1:
        # few distinct entropy levels -> one bin per (group of) level(s)
        edges = np.concatenate([uniq, uniq[-1:] + 1e-6])
    else:
        edges = np.unique(np.quantile(H, np.linspace(0, 1, n_q + 1)))
    res = {"entropy_edges": edges.tolist(), "bins": []}
    for q in range(len(edges) - 1):
        lo, hi = edges[q], edges[q + 1]
        last = q == len(edges) - 2
        mask = (H >= lo) & (H <= hi) if last else (H >= lo) & (H < hi)
        if mask.sum() == 0:
            continue
        row = {"q": q + 1, "H_lo": float(lo), "H_hi": float(hi), "n": int(mask.sum()), "ece": {}}
        for name, p in probs_by_method.items():
            row["ece"][name] = 100.0 * compute_ece_sampled(
                p[mask], soft_te[mask], n_bins=n_bins, n_samples=100, seed=seed)
        res["bins"].append(row)
    return res


# ──────────────────────────────────────────────────────────────────────────────
# D2: per-confidence-bin reliability gap (population form; no sampling needed)
#     gap_b = mean(conf | bin b) - mean(pi_pred | bin b)
# ──────────────────────────────────────────────────────────────────────────────

def diag_confidence_bins(probs_by_method: dict, soft_te: np.ndarray,
                         n_bins: int = 15) -> dict:
    edges = np.linspace(0, 1, n_bins + 1)
    res = {}
    for name, p in probs_by_method.items():
        conf = p.max(1)
        pred = p.argmax(1)
        hit = soft_te[np.arange(len(pred)), pred]   # E[1{y=pred}] = pi_pred
        rows = []
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            m = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
            if m.sum() == 0:
                continue
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
                         "conf": float(conf[m].mean()), "true_hit": float(hit[m].mean()),
                         "gap": float(conf[m].mean() - hit[m].mean())})
        ece = sum(r["n"] * abs(r["gap"]) for r in rows) / max(1, len(conf))
        res[name] = {"ece_true_pop": 100.0 * ece, "rows": rows}
    return res


# ──────────────────────────────────────────────────────────────────────────────
# D3: class- vs instance-structured ambiguity
# ──────────────────────────────────────────────────────────────────────────────

def diag_class_structure(probs_by_method: dict, soft_te: np.ndarray,
                         hard_te: np.ndarray, n_classes: int,
                         n_bins: int = 15, seed: int = 0) -> dict:
    """Per-(voted)class ECE_true for each method + eta^2 of entropy explained by class."""
    H = annotation_entropy(soft_te)
    # eta^2 (one-way): between-class SS / total SS
    grand = H.mean()
    ss_tot = float(((H - grand) ** 2).sum()) + 1e-12
    ss_between = 0.0
    for c in range(n_classes):
        m = hard_te == c
        if m.sum() > 0:
            ss_between += m.sum() * (H[m].mean() - grand) ** 2
    eta2 = float(ss_between / ss_tot)

    per_class = {}
    for name, p in probs_by_method.items():
        pc = {}
        for c in range(n_classes):
            m = hard_te == c
            if m.sum() >= 10:
                pc[c] = 100.0 * compute_ece_sampled(p[m], soft_te[m], n_bins=n_bins,
                                                     n_samples=50, seed=seed)
        per_class[name] = pc
    return {"eta2_entropy_by_class": eta2,
            "interpretation": ("class-structured (class-aware methods should help)"
                               if eta2 > 0.25 else
                               "instance-structured (class-aware methods unnecessary)"),
            "per_class_ece": per_class}


# ──────────────────────────────────────────────────────────────────────────────
# D4: LS-TS proxy validity  (A3)
# ──────────────────────────────────────────────────────────────────────────────

def diag_lsts_proxy(b: dict) -> dict:
    K = int(b["n_classes"])
    lc = torch.tensor(b["logits_cal"], dtype=torch.float32)
    p = torch.softmax(lc, 1).numpy()
    yh = b["hard_cal"].astype(int)
    soft = b["soft_cal"]
    idx = np.arange(len(yh))
    u = 1.0 - soft[idx, yh]          # true disagreement on voted class
    v = 1.0 - p[idx, yh]             # model complement-confidence
    rho = float(np.corrcoef(u, v)[0, 1]) if u.std() > 0 and v.std() > 0 else float("nan")
    # K-corrected proxy gap g = vbar*(1-1/K) - ubar  (Prop. lsts-valid (ii));
    # g>0 predicts over-smoothing (T_LS > T_SLTS).
    g = float(v.mean() * (1.0 - 1.0 / K) - u.mean())
    slts = SoftLabelTS().fit(lc, torch.tensor(soft, dtype=torch.float32))
    lsts = LabelSmoothTS().fit(lc, torch.tensor(yh, dtype=torch.long))
    return {"ubar": float(u.mean()), "vbar": float(v.mean()),
            "proxy_gap_raw_vbar_minus_ubar": float(v.mean() - u.mean()),
            "proxy_gap_K_corrected": g,
            "rho_u_v": rho, "T_slts": float(slts.T), "T_lsts": float(lsts.T),
            "predicted_oversmoothing": bool(g > 0)}


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def analyze_bundle(b: dict, n_bins: int = 15) -> dict:
    probs_by_method, temps = fit_all_methods(b)
    return {
        "name": str(b.get("name", "unnamed")),
        "n_classes": int(b["n_classes"]),
        "n_cal": int(len(b["hard_cal"])), "n_test": int(len(b["hard_te"])),
        "temperatures": temps,
        "D1_entropy_bins": diag_entropy_bins(probs_by_method, b["soft_te"], n_bins=n_bins),
        "D2_confidence_bins": diag_confidence_bins(probs_by_method, b["soft_te"], n_bins=n_bins),
        "D3_class_structure": diag_class_structure(probs_by_method, b["soft_te"],
                                                   b["hard_te"], int(b["n_classes"]), n_bins=n_bins),
        "D4_lsts_proxy": diag_lsts_proxy(b),
    }


def _print_report(r: dict) -> None:
    print(f"\n{'='*70}\n  DEPTH ANALYSIS: {r['name']}  "
          f"(K={r['n_classes']}, n_cal={r['n_cal']}, n_test={r['n_test']})\n{'='*70}")

    print("\n[D1] ECE_true (%) by annotation-entropy quintile  (low H -> high H):")
    methods = next((list(b["ece"].keys()) for b in r["D1_entropy_bins"]["bins"] if b["ece"]), [])
    hdr = "  bin  " + "".join(f"{m[:12]:>13}" for m in methods)
    print(hdr)
    for row in r["D1_entropy_bins"]["bins"]:
        cells = "".join(f"{row['ece'].get(m, float('nan')):>13.2f}" for m in methods)
        print(f"  Q{row['q']} n={row['n']:<5}{cells}")

    print("\n[D3] eta^2 (annotation entropy explained by class) = "
          f"{r['D3_class_structure']['eta2_entropy_by_class']:.3f}  "
          f"=> {r['D3_class_structure']['interpretation']}")

    d4 = r["D4_lsts_proxy"]
    print(f"\n[D4] LS-TS proxy:  ubar={d4['ubar']:.3f}  vbar={d4['vbar']:.3f}  "
          f"g(K-corrected)={d4['proxy_gap_K_corrected']:+.3f}  rho={d4['rho_u_v']:.3f}")
    print(f"     T_SLTS={d4['T_slts']:.2f}  T_LS-TS={d4['T_lsts']:.2f}  "
          f"-> predicted {'OVER' if d4['predicted_oversmoothing'] else 'UNDER'}-smoothing")


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic demo (verifies the analysis end-to-end without the logit cache)
# ──────────────────────────────────────────────────────────────────────────────

def make_demo_bundle(seed: int = 0) -> dict:
    """3-class problem with CONTINUOUS instance-level ambiguity.

    The middle cluster's annotator distribution interpolates from confident
    (pi~[0,1,0]) to maximally ambiguous (pi~[0,.5,.5]) with the x-coordinate, so
    annotation entropy varies continuously and D1's entropy bins are populated.
    Ambiguity is still class-structured (only class 1 is ambiguous) so D3 eta^2 is high.
    """
    rng = np.random.default_rng(seed)
    K, n = 3, 6000
    means = np.array([[-3.2, 1.1], [0.0, 0.0], [3.2, -1.1]])
    Xs, soft, hard = [], [], []
    m = n // 3
    # class 0 and class 2: unambiguous
    for c in (0, 2):
        x = rng.normal(means[c], [0.7, 0.55], size=(m, 2))
        pi = np.zeros((m, K)); pi[:, c] = 1.0
        Xs.append(x); soft.append(pi); hard.append(np.full(m, c))
    # class 1: ambiguity grows with sigmoid of x[:,0]
    x1 = rng.normal(means[1], [1.15, 0.75], size=(m, 2))
    t = 1.0 / (1.0 + np.exp(-1.2 * x1[:, 0]))         # in (0,1)
    minority = 0.5 * t                                  # mass shifted to class 2, up to 0.5
    pi1 = np.zeros((m, K)); pi1[:, 1] = 1.0 - minority; pi1[:, 2] = minority
    Xs.append(x1); soft.append(pi1); hard.append(np.ones(m, dtype=int))
    X = np.concatenate(Xs); soft = np.concatenate(soft); hard = np.concatenate(hard).astype(int)
    W = np.array([[-2.0, 1.5], [0.0, 0.0], [2.0, -1.5]])
    logits = (X @ W.T) * 1.8                            # deliberately overconfident
    idx = rng.permutation(len(X)); half = len(X) // 2
    cal, te = idx[:half], idx[half:]
    return {"name": "DEMO (synthetic ambiguous)", "n_classes": K,
            "logits_cal": logits[cal], "logits_te": logits[te],
            "soft_cal": soft[cal], "soft_te": soft[te],
            "hard_cal": hard[cal], "hard_te": hard[te]}


def load_bundle(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    b = {k: z[k] for k in z.files}
    b["name"] = str(b["name"]) if "name" in b else Path(path).stem
    b["n_classes"] = int(b["n_classes"])
    return b


def _np(a):
    """Coerce a torch tensor or array-like to a float/int numpy array."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def save_bundle(path, name, n_classes,
                logits_cal, logits_te, soft_cal, soft_te, hard_cal, hard_te):
    """Persist a depth-analysis bundle. Call this from a run_*.py script with the
    exact cal/test arrays it already computed (see BUNDLES.md). Accepts numpy or
    torch tensors. This guarantees the bundle matches the published split/logits."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, name=str(name), n_classes=int(n_classes),
             logits_cal=_np(logits_cal).astype(np.float32),
             logits_te=_np(logits_te).astype(np.float32),
             soft_cal=_np(soft_cal).astype(np.float32),
             soft_te=_np(soft_te).astype(np.float32),
             hard_cal=_np(hard_cal).astype(np.int64),
             hard_te=_np(hard_te).astype(np.int64))
    print(f"  bundle saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run on a synthetic dataset (no cache needed)")
    ap.add_argument("--bundle", help="path to a single .npz bundle")
    ap.add_argument("--bundle-dir", help="directory of .npz bundles (analyze all)")
    ap.add_argument("--out", default="results/depth", help="output dir for JSON reports")
    ap.add_argument("--n-bins", type=int, default=15)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    bundles = []
    if args.demo:
        bundles = [make_demo_bundle()]
    elif args.bundle:
        bundles = [load_bundle(args.bundle)]
    elif args.bundle_dir:
        bundles = [load_bundle(str(p)) for p in sorted(Path(args.bundle_dir).glob("*.npz"))]
    else:
        ap.error("specify --demo, --bundle, or --bundle-dir")

    for b in bundles:
        r = analyze_bundle(b, n_bins=args.n_bins)
        _print_report(r)
        tag = "demo" if args.demo else Path(b["name"]).stem.replace(" ", "_")
        with open(Path(args.out) / f"depth_{tag}.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"\n  saved -> {Path(args.out) / f'depth_{tag}.json'}")


if __name__ == "__main__":
    main()
