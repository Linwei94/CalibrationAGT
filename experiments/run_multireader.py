"""
Calibration under REAL multi-reader ground truth (TPAMI resubmission).

Addresses Reviewer 1 W4 / AE #2: validate ambiguity-aware calibration on a dataset
with TRUE per-image annotations from multiple independent readers, rather than the
class-conditional synthetic annotators used for ISIC/DermaMNIST.

This harness is dataset-agnostic. It consumes individual per-image reader labels
(the realistic multi-reader regime -> directly drives MCTS), builds the empirical
annotator distribution and the majority-voted label, fits every calibrator from
calibration.py, and reports the same metrics as the main experiments.

Two-stage design so the calibration logic is verifiable without the (large) medical
data or a trained model:

  Stage A (dataset+model specific): produce a prepared .npz with
      logits_cal (Ncal,K), logits_test (Nt,K),
      ann_cal (Ncal,R int, -1 = missing), ann_test (Nt,R int, -1 = missing),
      n_classes.
    For VinDr-CXR / CheXpert / LIDC-IDRI use the loader stubs below (each reuses
    the same backbone/extract_logits pattern as run_isic2019.py); R = #readers
    (3 for VinDr-CXR/CheXpert val, 4 for LIDC-IDRI nodules).

  Stage B (generic, runnable here): calibrate + evaluate from that .npz.

USAGE
-----
  python run_multireader.py --demo                 # synthetic instance-level readers; verifies harness
  python run_multireader.py --data prepared/lidc_resnet.npz --name lidc_resnet18
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    TemperatureScaling, PlattScaling, DirichletCalibration, SoftLabelTS,
    MonteCarloTS, VectorScaling, SoftPlattScaling, LabelSmoothTS,
    SoftIsotonicRegression, apply_parametric,
)
from metrics import compute_all_metrics, annotation_entropy, print_results_table


# ──────────────────────────────────────────────────────────────────────────────
# Annotation aggregation: ragged reader labels -> soft distribution + voted label
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(ann: np.ndarray, K: int):
    """ann: (N, R) int with -1 for missing readers. Returns (soft (N,K), voted (N,)).

    soft_i = empirical reader frequency; voted_i = majority reader label (ties -> lowest).
    Images with >=1 valid reader are kept.
    """
    N = ann.shape[0]
    soft = np.zeros((N, K), np.float32)
    for i in range(N):
        valid = ann[i][ann[i] >= 0]
        if len(valid) == 0:
            soft[i] = 1.0 / K
            continue
        soft[i] = np.bincount(valid, minlength=K).astype(np.float32) / len(valid)
    voted = soft.argmax(1).astype(np.int64)
    return soft, voted


# ──────────────────────────────────────────────────────────────────────────────
# Stage B: calibrate + evaluate  (generic, reuses calibration.py + metrics.py)
# ──────────────────────────────────────────────────────────────────────────────

def run(logits_cal, ann_cal, logits_test, ann_test, K, name, n_bins=15):
    soft_cal, voted_cal = aggregate(ann_cal, K)
    soft_te,  voted_te  = aggregate(ann_test, K)
    lc = torch.tensor(logits_cal, dtype=torch.float32)
    lt = logits_test
    yh = torch.tensor(voted_cal, dtype=torch.long)
    ys = torch.tensor(soft_cal, dtype=torch.float32)
    probs_cal = torch.softmax(lc, 1).numpy()
    probs_te  = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()

    print(f"\n[{name}]  K={K}  n_cal={len(voted_cal)}  n_test={len(voted_te)}  "
          f"mean readers/img(cal)={float((ann_cal>=0).sum(1).mean()):.1f}  "
          f"mean H_test={annotation_entropy(soft_te).mean():.3f}")

    # voted-label baselines
    ts   = TemperatureScaling().fit(lc, yh)
    ps   = PlattScaling(K).fit(lc, yh)
    dc_h = DirichletCalibration(K).fit_hard(lc, yh)
    lsts = LabelSmoothTS().fit(lc, yh)                 # annotation-free
    # ambiguity-aware (soft / individual annotations)
    slts = SoftLabelTS().fit(lc, ys)
    mcts1 = MonteCarloTS(n_samples=1).fit(lc, ys)      # single reader per image
    vs   = VectorScaling(K).fit(lc, ys)
    sp   = SoftPlattScaling(K).fit(lc, ys)
    dc_s = DirichletCalibration(K).fit_soft(lc, ys)
    ir   = SoftIsotonicRegression().fit(probs_cal, soft_cal)

    results = []
    para = [("Uncalibrated", probs_te), ("TS", apply_parametric(ts, lt)),
            ("Platt (PS)", apply_parametric(ps, lt)), ("Dirichlet-Hard", apply_parametric(dc_h, lt)),
            ("LS-TS", apply_parametric(lsts, lt)), ("SLTS", apply_parametric(slts, lt)),
            ("MCTS S=1", apply_parametric(mcts1, lt)), ("VS", apply_parametric(vs, lt)),
            ("SoftPlatt", apply_parametric(sp, lt)), ("Dirichlet-Soft", apply_parametric(dc_s, lt))]
    for nm, p in para:
        r = compute_all_metrics(p, voted_te, soft_te, n_bins=n_bins, name=nm)
        r["temperature"] = {"TS": ts.T, "SLTS": slts.T, "MCTS S=1": mcts1.T, "LS-TS": lsts.T}.get(nm)
        results.append(r)
    # IR-Soft (non-parametric, top-class)
    cal_conf, pred = ir.calibrate(probs_te)
    pir = np.full_like(probs_te, 0.0)
    for i in range(len(probs_te)):
        pir[i, pred[i]] = cal_conf[i]
        rest = np.ones(K) * (1 - cal_conf[i]) / (K - 1); rest[pred[i]] = 0
        pir[i] += rest
    r = compute_all_metrics(pir, voted_te, soft_te, n_bins=n_bins, name="IR-Soft"); r["temperature"] = None
    results.append(r)

    print_results_table(results)
    return {"name": name, "n_classes": K, "n_cal": int(len(voted_cal)),
            "n_test": int(len(voted_te)), "main_results": results}


# ──────────────────────────────────────────────────────────────────────────────
# Stage A loaders (STUBS — fill in on the machine with the data + a trained model).
# Each must return (logits_cal, ann_cal, logits_test, ann_test, K).
# The model/logit-extraction mirrors run_isic2019.py exactly; only the labels differ.
# ──────────────────────────────────────────────────────────────────────────────

def load_lidc_idri(cache_dir):
    """LIDC-IDRI lung nodules: 4 radiologists rate malignancy 1-5 per nodule.
    Use `pylidc` to enumerate nodules with >=3 readers; map ratings -> K classes
    (e.g. K=2: benign {1,2} vs malignant {4,5}, drop 3; or K=5 ordinal).
    ann[i] = the per-reader class labels for nodule i (pad to R=4 with -1).
    Train a small CNN on majority labels (train split), cache val/test logits as
    in run_isic2019.extract_logits, then split val->calibration."""
    raise NotImplementedError(
        "LIDC-IDRI loader: install pylidc, build nodule crops + 4-reader labels, "
        "train backbone, extract logits. Returns (logits_cal, ann_cal, logits_test, ann_test, K). "
        "See run_isic2019.py for the backbone/extract_logits/split pattern.")

def load_vindr_cxr(cache_dir):
    """VinDr-CXR: 3 independent radiologists per image (image-level findings).
    Pick a multi-class label (e.g. the global diagnosis field); ann[i] = the 3
    reader labels. Same training/extract/split pattern as run_isic2019.py."""
    raise NotImplementedError("VinDr-CXR loader stub — see docstring and run_isic2019.py.")

def load_chexpert(cache_dir):
    """CheXpert: validation set has 3 board-certified radiologist labelings.
    Use those 3 as the reader annotations on the calibration/eval split; train on
    the (large) train split's majority labels."""
    raise NotImplementedError("CheXpert loader stub — see docstring and run_isic2019.py.")

LOADERS = {"lidc": load_lidc_idri, "vindr": load_vindr_cxr, "chexpert": load_chexpert}


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic demo: INSTANCE-LEVEL reader disagreement (unlike class-conditional ISIC)
# ──────────────────────────────────────────────────────────────────────────────

def make_demo(seed=0, K=4, n=4000, R=4):
    """Each image has a latent difficulty d_i; readers flip to a random other class
    with prob d_i. Disagreement is per-image (not per-class), so this is the
    instance-level regime the synthetic medical model could NOT capture."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)                       # true class
    d = rng.beta(1.5, 4.0, size=n)                       # per-image difficulty in (0,1)
    ann = np.full((n, R), -1, np.int64)
    for i in range(n):
        for r in range(R):
            if rng.random() < d[i]:
                ann[i, r] = rng.integers(0, K)           # confused reader -> random class
            else:
                ann[i, r] = y[i]
    # a "model": logit mass on true class scaled by (1-d), plus noise -> overconfident
    logits = rng.normal(0, 0.3, size=(n, K))
    logits[np.arange(n), y] += 3.0 * (1 - 0.5 * d)
    idx = rng.permutation(n); h = n // 2
    cal, te = idx[:h], idx[h:]
    return logits[cal], ann[cal], logits[te], ann[te], K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--dataset", choices=list(LOADERS), help="real dataset (needs data + trained model)")
    ap.add_argument("--data", help="prepared .npz (logits_cal, ann_cal, logits_test, ann_test, n_classes)")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--name", default="multireader")
    ap.add_argument("--n-bins", type=int, default=15)
    args = ap.parse_args()

    if args.demo:
        lc, ac, lt, at, K = make_demo()
        name = "demo_multireader"
    elif args.data:
        z = np.load(args.data, allow_pickle=True)
        lc, ac, lt, at, K = z["logits_cal"], z["ann_cal"], z["logits_test"], z["ann_test"], int(z["n_classes"])
        name = args.name
    elif args.dataset:
        lc, ac, lt, at, K = LOADERS[args.dataset](args.cache_dir)
        name = args.name if args.name != "multireader" else args.dataset
    else:
        ap.error("specify --demo, --data, or --dataset")

    out = run(lc, ac, lt, at, K, name, n_bins=args.n_bins)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    p = Path(args.results_dir) / f"multireader_{name}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
