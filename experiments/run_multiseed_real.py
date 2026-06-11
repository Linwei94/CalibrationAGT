"""
Multi-seed robustness for the existing real-data bundles (TPAMI resubmission).

For each persisted real-data bundle we:
  1. concatenate the calibration and test halves (logits, soft, hard) back into the
     full pool the bundle was built from;
  2. for seeds 42..46, draw a fresh stratified (by voted/hard label) 50/50 split;
  3. refit each calibrator on the new calibration half:
       TS (voted hard labels), SLTS, MCTS S=1, Dirichlet-Soft (skipped if K>50),
       IR-Soft;
  4. compute ECE_true on the new test half as the POPULATION reliability gap,
     identical to analyze_depth.diag_confidence_bins -> "ece_true_pop":
       per 15-equal-width confidence bin b,
         gap_b = mean(conf | b) - mean(soft[arange, pred] | b)
       ECE_true = sum_b n_b * |gap_b| / N    (then *100 for %).

We report, per bundle, the mean +/- std of ECE_true (%) over the 5 seeds for each method.

Bundles:
  /root/CalibrationAGT/experiments/results/real/{cifar10h_vit_real,cifar10h_cnn_real,
      dermamnist_resnet18,dermamnist_vit_s16,nli_llm_real}.npz
  /tmp/real3/bundles/imagenet_real_resnet50.npz
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/CalibrationAGT/experiments")
from calibration import (
    TemperatureScaling, SoftLabelTS, MonteCarloTS,
    DirichletCalibration, SoftIsotonicRegression, apply_parametric,
)

BUNDLES = [
    "/root/CalibrationAGT/experiments/results/real/cifar10h_vit_real.npz",
    "/root/CalibrationAGT/experiments/results/real/cifar10h_cnn_real.npz",
    "/root/CalibrationAGT/experiments/results/real/dermamnist_resnet18.npz",
    "/root/CalibrationAGT/experiments/results/real/dermamnist_vit_s16.npz",
    "/root/CalibrationAGT/experiments/results/real/nli_llm_real.npz",
    "/tmp/real3/bundles/imagenet_real_resnet50.npz",
]

SEEDS = [42, 43, 44, 45, 46]
N_BINS = 15
DIRICHLET_MAX_K = 50  # skip Dirichlet-Soft when K > DIRICHLET_MAX_K


def load_pool(path):
    """Load a bundle and concatenate cal+test into a single pool."""
    z = np.load(path, allow_pickle=True)
    name = str(z["name"])
    K = int(z["n_classes"])
    logits = np.concatenate([z["logits_cal"], z["logits_te"]], axis=0).astype(np.float32)
    soft = np.concatenate([z["soft_cal"], z["soft_te"]], axis=0).astype(np.float32)
    hard = np.concatenate([z["hard_cal"], z["hard_te"]], axis=0).astype(np.int64)
    return name, K, logits, soft, hard


def stratified_split(hard, K, seed):
    """Stratified (by voted/hard label) 50/50 split. Mirrors run_multiseed's
    make_split_cifar10h: per-class permutation, first half -> cal, rest -> test."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(hard))
    idx_cal_parts, idx_te_parts = [], []
    for k in range(K):
        arr = idx[hard == k]
        if len(arr) == 0:
            continue
        perm = rng.permutation(arr)
        cut = len(arr) // 2
        idx_cal_parts.append(perm[:cut])
        idx_te_parts.append(perm[cut:])
    idx_cal = np.concatenate(idx_cal_parts)
    idx_te = np.concatenate(idx_te_parts)
    idx_cal.sort(); idx_te.sort()
    return idx_cal, idx_te


def fit_methods(K, logits_cal, soft_cal, hard_cal, logits_te):
    """Fit the requested calibrators on the cal half; return {method: probs_te}.
    Mirrors analyze_depth.fit_all_methods for the requested subset."""
    lc = torch.tensor(logits_cal, dtype=torch.float32)
    yh = torch.tensor(hard_cal, dtype=torch.long)
    ys = torch.tensor(soft_cal, dtype=torch.float32)
    lt = logits_te
    probs_cal_np = torch.softmax(lc, dim=1).numpy()

    out = {}

    ts = TemperatureScaling().fit(lc, yh)
    out["TS"] = apply_parametric(ts, lt)

    slts = SoftLabelTS().fit(lc, ys)
    out["SLTS"] = apply_parametric(slts, lt)

    mcts1 = MonteCarloTS(n_samples=1).fit(lc, ys)
    out["MCTS S=1"] = apply_parametric(mcts1, lt)

    if K <= DIRICHLET_MAX_K:
        dcs = DirichletCalibration(K).fit_soft(lc, ys)
        out["Dirichlet-Soft"] = apply_parametric(dcs, lt)

    # IR-Soft: monotone top-class confidence remap, spread (1-conf) over other K-1
    ir = SoftIsotonicRegression().fit(probs_cal_np, soft_cal)
    probs_te_np = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()
    cal_conf, pred = ir.calibrate(probs_te_np)
    pir = np.zeros_like(probs_te_np)
    for i in range(len(probs_te_np)):
        pir[i, pred[i]] = cal_conf[i]
        rest = np.ones(K) * (1 - cal_conf[i]) / (K - 1)
        rest[pred[i]] = 0
        pir[i] += rest
    out["IR-Soft"] = pir

    return out


def ece_true_pop(probs, soft_te, n_bins=N_BINS):
    """Population reliability-gap ECE_true (%), identical to
    analyze_depth.diag_confidence_bins[...]['ece_true_pop'].

      gap_b = mean(conf|b) - mean(soft[arange,pred]|b)
      ECE   = sum_b n_b * |gap_b| / N
    """
    edges = np.linspace(0, 1, n_bins + 1)
    conf = probs.max(1)
    pred = probs.argmax(1)
    hit = soft_te[np.arange(len(pred)), pred]   # pi_pred
    total = 0.0
    N = max(1, len(conf))
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
        nb = int(m.sum())
        if nb == 0:
            continue
        gap = conf[m].mean() - hit[m].mean()
        total += nb * abs(gap)
    return 100.0 * total / N


def main():
    summary = {}
    for path in BUNDLES:
        name, K, logits, soft, hard = load_pool(path)
        print(f"\n{'='*60}\n  {name}  (K={K}, N_pool={len(hard)})\n{'='*60}")
        per_seed = {}  # method -> list of ECE_true over seeds
        for seed in SEEDS:
            idx_cal, idx_te = stratified_split(hard, K, seed)
            probs_by_method = fit_methods(
                K, logits[idx_cal], soft[idx_cal], hard[idx_cal], logits[idx_te])
            soft_te = soft[idx_te]
            row = []
            for method, p in probs_by_method.items():
                e = ece_true_pop(p, soft_te)
                per_seed.setdefault(method, []).append(e)
                row.append(f"{method}={e:.2f}")
            print(f"  seed {seed}: " + "  ".join(row))

        bundle_summary = {}
        for method, vals in per_seed.items():
            arr = np.array(vals, dtype=float)
            bundle_summary[method] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),  # population std (ddof=0), matches run_multiseed
                "per_seed": [float(v) for v in vals],
            }
        summary[name] = {
            "n_classes": K,
            "n_pool": int(len(hard)),
            "seeds": SEEDS,
            "dirichlet_skipped": K > DIRICHLET_MAX_K,
            "methods": bundle_summary,
        }

        print(f"\n  mean +/- std ECE_true (%) over seeds {SEEDS}:")
        for method, s in bundle_summary.items():
            print(f"    {method:<16} {s['mean']:6.2f} +/- {s['std']:.2f}")

    out_path = Path("/tmp/real4/multiseed_real.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metric": "ece_true_pop (analyze_depth D2)",
                   "n_bins": N_BINS,
                   "split": "stratified-by-hard 50/50",
                   "summary": summary}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
