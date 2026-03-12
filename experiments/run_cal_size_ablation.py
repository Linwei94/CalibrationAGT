"""
Calibration-set-size ablation for ambiguity-aware calibrators.

Fixes annotation count at the full amount, and varies the fraction of
calibration data used (10%..100%). Measures how many calibration examples
are needed for each method to converge.

Datasets supported: cifar10h, chaosnli
Methods: TS, SLTS, VS, Dirichlet-Soft, IR-Soft

Usage
-----
    python run_cal_size_ablation.py --dataset cifar10h --arch resnet50
    python run_cal_size_ablation.py --dataset cifar10h --arch vit_b16
    python run_cal_size_ablation.py --dataset chaosnli --arch roberta_large
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    DirichletCalibration,
    LabelSmoothTS,
    SoftIsotonicRegression,
    SoftLabelTS,
    TemperatureScaling,
    VectorScaling,
)
from metrics import compute_all_metrics
from run_cifar10h import download_cifar10h, get_cifar10_testset
from run_chaosnli import download_chaosnli, parse_chaosnli


METHOD_ORDER = ["TS", "LS-TS", "SLTS", "VS", "Dirichlet-Soft", "IR-Soft"]

PLOT_COLORS = {
    "TS": "#777777",
    "SLTS": "#4CAF80",
    "VS": "#F39C12",
    "Dirichlet-Soft": "#D45B3A",
    "LS-TS": "#9C27B0",
    "IR-Soft": "#34495E",
}

DEFAULT_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def stratified_subsample(
    idx_cal: np.ndarray,
    hard_labels: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified subsample of n indices from idx_cal, maintaining class proportions."""
    classes = np.unique(hard_labels[idx_cal])
    n_per_class = {}
    total_assigned = 0
    for c in classes:
        c_idx = idx_cal[hard_labels[idx_cal] == c]
        frac = len(c_idx) / len(idx_cal)
        n_c = max(1, round(n * frac))
        n_c = min(n_c, len(c_idx))
        n_per_class[c] = n_c
        total_assigned += n_c

    # Trim to exactly n if rounding over-allocated
    while total_assigned > n:
        for c in classes:
            if n_per_class[c] > 1 and total_assigned > n:
                n_per_class[c] -= 1
                total_assigned -= 1

    parts = []
    for c in classes:
        c_idx = idx_cal[hard_labels[idx_cal] == c]
        chosen = rng.choice(c_idx, size=n_per_class[c], replace=False)
        parts.append(chosen)
    return np.concatenate(parts)


def fit_and_eval_subset(
    method: str,
    logits_cal_sub: np.ndarray,
    labels_hard_cal_sub: np.ndarray,
    probs_cal_sub: np.ndarray,
    labels_soft_cal_sub: np.ndarray,
    logits_te: np.ndarray,
    labels_hard_te: np.ndarray,
    labels_soft_te: np.ndarray,
    n_bins: int,
    adam_epochs: int,
    device: torch.device,
) -> dict:
    logits_cal_t = torch.tensor(logits_cal_sub, dtype=torch.float32, device=device)
    labels_soft_cal_t = torch.tensor(labels_soft_cal_sub, dtype=torch.float32, device=device)
    labels_hard_cal_t = torch.tensor(labels_hard_cal_sub, dtype=torch.long, device=device)

    def apply_cal(calibrator, logits_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(logits_np, dtype=torch.float32, device=device)
            p = torch.softmax(calibrator(t), dim=1)
        return p.detach().cpu().numpy()

    if method == "TS":
        cal = TemperatureScaling().to(device).fit(logits_cal_t, labels_hard_cal_t)
        probs = apply_cal(cal, logits_te)
        temperature = cal.T
    elif method == "LS-TS":
        cal = LabelSmoothTS().to(device).fit(logits_cal_t, labels_hard_cal_t)
        probs = apply_cal(cal, logits_te)
        temperature = cal.T
    elif method == "SLTS":
        cal = SoftLabelTS().to(device).fit(logits_cal_t, labels_soft_cal_t)
        probs = apply_cal(cal, logits_te)
        temperature = cal.T
    elif method == "VS":
        cal = VectorScaling(logits_cal_sub.shape[1]).to(device).fit(logits_cal_t, labels_soft_cal_t, n_epochs=adam_epochs)
        probs = apply_cal(cal, logits_te)
        temperature = None
    elif method == "Dirichlet-Soft":
        cal = DirichletCalibration(logits_cal_sub.shape[1], n_epochs=adam_epochs).to(device).fit_soft(logits_cal_t, labels_soft_cal_t)
        probs = apply_cal(cal, logits_te)
        temperature = None
    elif method == "IR-Soft":
        ir = SoftIsotonicRegression().fit(probs_cal_sub, labels_soft_cal_sub)
        conf, pred = ir.calibrate(torch.softmax(torch.tensor(logits_te), dim=1).numpy())
        probs = np.zeros((len(pred), logits_cal_sub.shape[1]), dtype=np.float64)
        for i in range(len(pred)):
            probs[i, pred[i]] = conf[i]
            rest = (1.0 - conf[i]) / (probs.shape[1] - 1)
            probs[i] += rest
            probs[i, pred[i]] = conf[i]
        temperature = None
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics = compute_all_metrics(probs, labels_hard_te, labels_soft_te, n_bins=n_bins, name=method)
    metrics["temperature"] = temperature
    metrics["ece_true"] = metrics["ece_sampled"]
    metrics["n_cal"] = int(len(logits_cal_sub))
    return metrics


def load_cifar10h_data(cache_dir: Path, arch: str, seed: int):
    from run_cifar10h import download_cifar10h, get_cifar10_testset
    labels_soft_all = download_cifar10h(str(cache_dir))
    logits_all = np.load(cache_dir / f"logits_test_{arch}.npy")
    testset = get_cifar10_testset(str(cache_dir / "cifar10"))
    labels_hard_all = np.array([y for _, y in testset])

    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels_soft_all))
    K = labels_soft_all.shape[1]
    per_class = [idx[labels_hard_all == k] for k in range(K)]
    idx_cal_parts, idx_te_parts = [], []
    for arr in per_class:
        perm = rng.permutation(arr)
        cut = len(arr) // 2
        idx_cal_parts.append(perm[:cut])
        idx_te_parts.append(perm[cut:])
    idx_cal = np.concatenate(idx_cal_parts)
    idx_te = np.concatenate(idx_te_parts)
    idx_cal.sort()
    idx_te.sort()

    return (
        logits_all, labels_hard_all, labels_soft_all,
        idx_cal, idx_te, rng,
    )


def load_chaosnli_data(cache_dir: Path, arch: str, seed: int):
    raw_data = download_chaosnli(str(cache_dir), subset="snli") + \
               download_chaosnli(str(cache_dir), subset="mnli")
    _, soft_labels_all, hard_labels_all = parse_chaosnli(raw_data)
    N = len(hard_labels_all)

    logits_all = np.load(cache_dir / f"logits_chaosnli_combined_{arch}.npy")

    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(3):
        c_idx = idx[hard_labels_all == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True
    idx_cal = idx[cal_mask]
    idx_te = idx[~cal_mask]

    return (
        logits_all, hard_labels_all, soft_labels_all,
        idx_cal, idx_te, rng,
    )


def make_plot(results: dict, out_pdf: Path, out_png: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    fractions = results["fractions"]
    x_pct = [f * 100 for f in fractions]

    for method in METHOD_ORDER:
        if method not in results["methods"]:
            continue
        trace = results["methods"][method]
        y = [trace[str(f)]["ece_true"] * 100 for f in fractions]
        ls = "--" if method == "TS" else "-"
        ax.plot(x_pct, y, label=method, color=PLOT_COLORS[method], lw=2.0, ls=ls)

    ax.set_xlabel("Calibration set size (%)")
    ax.set_ylabel("ECE_true (%)")
    ax.set_xticks([int(f * 100) for f in fractions])
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10h", "chaosnli"], default="cifar10h")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    p.add_argument("--cache-dir", default="experiments/cache")
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--figures-dir", default="experiments/figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--adam-epochs", type=int, default=400)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.dataset == "cifar10h":
        logits_all, hard_labels_all, soft_labels_all, idx_cal, idx_te, rng = \
            load_cifar10h_data(cache_dir, args.arch, args.seed)
    else:
        logits_all, hard_labels_all, soft_labels_all, idx_cal, idx_te, rng = \
            load_chaosnli_data(cache_dir, args.arch, args.seed)

    logits_te = logits_all[idx_te]
    labels_hard_te = hard_labels_all[idx_te]
    labels_soft_te = soft_labels_all[idx_te]

    print(f"Full cal n={len(idx_cal)}, Test n={len(idx_te)}")

    # ── Main ablation loop ────────────────────────────────────────────────────
    fractions = sorted(args.fractions)
    results = {
        "dataset": args.dataset,
        "arch": args.arch,
        "seed": args.seed,
        "fractions": fractions,
        "n_cal_full": int(len(idx_cal)),
        "n_test": int(len(idx_te)),
        "methods": {method: {} for method in METHOD_ORDER},
    }

    for frac in fractions:
        n_sub = max(len(np.unique(hard_labels_all[idx_cal])), int(frac * len(idx_cal)))
        idx_sub = stratified_subsample(idx_cal, hard_labels_all, n_sub, rng)

        logits_cal_sub = logits_all[idx_sub]
        labels_hard_cal_sub = hard_labels_all[idx_sub]
        probs_cal_sub = torch.softmax(torch.tensor(logits_cal_sub), dim=1).numpy()
        labels_soft_cal_sub = soft_labels_all[idx_sub]

        print(f"\n=== frac={frac:.0%}  n_cal={len(idx_sub)} ===")
        for method in METHOD_ORDER:
            metrics = fit_and_eval_subset(
                method=method,
                logits_cal_sub=logits_cal_sub,
                labels_hard_cal_sub=labels_hard_cal_sub,
                probs_cal_sub=probs_cal_sub,
                labels_soft_cal_sub=labels_soft_cal_sub,
                logits_te=logits_te,
                labels_hard_te=labels_hard_te,
                labels_soft_te=labels_soft_te,
                n_bins=args.n_bins,
                adam_epochs=args.adam_epochs,
                device=device,
            )
            results["methods"][method][str(frac)] = metrics
            print(f"  {method:<15} ECE_true={metrics['ece_true']*100:5.2f}%  n_cal={metrics['n_cal']}")

    suffix = f"_{args.tag}" if args.tag else ""
    out_json = results_dir / f"cal_size_ablation_{args.dataset}_{args.arch}{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {out_json}")

    out_pdf = figures_dir / f"cal_size_ablation_{args.dataset}_{args.arch}{suffix}.pdf"
    out_png = figures_dir / f"cal_size_ablation_{args.dataset}_{args.arch}{suffix}.png"
    make_plot(results, out_pdf, out_png)
    print(f"Saved plot → {out_pdf}")


if __name__ == "__main__":
    main()
