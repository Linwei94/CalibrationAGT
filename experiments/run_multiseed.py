"""
Multi-seed statistical significance analysis.

Re-runs calibration with 5 different random seeds (affecting the cal/test split)
and reports mean ± std of ECE_true for key methods.

Datasets: cifar10h, chaosnli
Methods: TS, SLTS, Dirichlet-Soft, IR-Soft

Usage
-----
    python run_multiseed.py --dataset cifar10h --arch resnet50
    python run_multiseed.py --dataset chaosnli --arch roberta_large
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
    SoftIsotonicRegression,
    SoftLabelTS,
    TemperatureScaling,
)
from metrics import compute_all_metrics
from run_chaosnli import download_chaosnli, parse_chaosnli
from run_cifar10h import download_cifar10h, get_cifar10_testset


METHOD_ORDER = ["TS", "SLTS", "Dirichlet-Soft", "IR-Soft"]

PLOT_COLORS = {
    "TS": "#777777",
    "SLTS": "#4CAF80",
    "Dirichlet-Soft": "#D45B3A",
    "IR-Soft": "#34495E",
}

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def make_split_cifar10h(logits_all, hard_labels_all, soft_labels_all, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(hard_labels_all))
    K = soft_labels_all.shape[1]
    per_class = [idx[hard_labels_all == k] for k in range(K)]
    idx_cal_parts, idx_te_parts = [], []
    for arr in per_class:
        perm = rng.permutation(arr)
        cut = len(arr) // 2
        idx_cal_parts.append(perm[:cut])
        idx_te_parts.append(perm[cut:])
    idx_cal = np.concatenate(idx_cal_parts)
    idx_te = np.concatenate(idx_te_parts)
    idx_cal.sort(); idx_te.sort()
    return idx_cal, idx_te


def make_split_chaosnli(hard_labels_all, N, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(3):
        c_idx = idx[hard_labels_all == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True
    return idx[cal_mask], idx[~cal_mask]


def fit_and_eval_method(
    method: str,
    logits_cal: np.ndarray,
    probs_cal: np.ndarray,
    labels_hard_cal: np.ndarray,
    labels_soft_cal: np.ndarray,
    logits_te: np.ndarray,
    labels_hard_te: np.ndarray,
    labels_soft_te: np.ndarray,
    n_bins: int,
    adam_epochs: int,
    device: torch.device,
) -> dict:
    logits_cal_t = torch.tensor(logits_cal, dtype=torch.float32, device=device)
    labels_soft_cal_t = torch.tensor(labels_soft_cal, dtype=torch.float32, device=device)
    labels_hard_cal_t = torch.tensor(labels_hard_cal, dtype=torch.long, device=device)

    def apply_cal(calibrator, logits_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(logits_np, dtype=torch.float32, device=device)
            p = torch.softmax(calibrator(t), dim=1)
        return p.detach().cpu().numpy()

    if method == "TS":
        cal = TemperatureScaling().to(device).fit(logits_cal_t, labels_hard_cal_t)
        probs = apply_cal(cal, logits_te)
        temperature = cal.T
    elif method == "SLTS":
        cal = SoftLabelTS().to(device).fit(logits_cal_t, labels_soft_cal_t)
        probs = apply_cal(cal, logits_te)
        temperature = cal.T
    elif method == "Dirichlet-Soft":
        cal = DirichletCalibration(logits_cal.shape[1], n_epochs=adam_epochs).to(device).fit_soft(logits_cal_t, labels_soft_cal_t)
        probs = apply_cal(cal, logits_te)
        temperature = None
    elif method == "IR-Soft":
        ir = SoftIsotonicRegression().fit(probs_cal, labels_soft_cal)
        conf, pred = ir.calibrate(torch.softmax(torch.tensor(logits_te), dim=1).numpy())
        probs = np.zeros((len(pred), logits_cal.shape[1]), dtype=np.float64)
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
    return metrics


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

    summary = results["summary"]
    methods = [m for m in METHOD_ORDER if m in summary]
    means = [summary[m]["mean"] * 100 for m in methods]
    stds = [summary[m]["std"] * 100 for m in methods]
    colors = [PLOT_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.55,
                  capsize=5, error_kw={"lw": 1.8}, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("ECE_true (%)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10h", "chaosnli"], default="cifar10h")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--cache-dir", default="experiments/cache")
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--figures-dir", default="experiments/figures")
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

    # ── Load full data once ───────────────────────────────────────────────────
    if args.dataset == "cifar10h":
        soft_labels_all = download_cifar10h(str(cache_dir))
        logits_all = np.load(cache_dir / f"logits_test_{args.arch}.npy")
        testset = get_cifar10_testset(str(cache_dir / "cifar10"))
        hard_labels_all = np.array([y for _, y in testset])
    else:
        raw_data = download_chaosnli(str(cache_dir), subset="snli") + \
                   download_chaosnli(str(cache_dir), subset="mnli")
        _, soft_labels_all, hard_labels_all = parse_chaosnli(raw_data)
        logits_all = np.load(cache_dir / f"logits_chaosnli_combined_{args.arch}.npy")

    N = len(hard_labels_all)

    # ── Multi-seed loop ───────────────────────────────────────────────────────
    per_seed: dict[int, dict[str, float]] = {}

    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"  Seed = {seed}")
        print(f"{'='*50}")

        if args.dataset == "cifar10h":
            idx_cal, idx_te = make_split_cifar10h(logits_all, hard_labels_all, soft_labels_all, seed)
        else:
            idx_cal, idx_te = make_split_chaosnli(hard_labels_all, N, seed)

        logits_cal = logits_all[idx_cal]
        logits_te = logits_all[idx_te]
        probs_cal = torch.softmax(torch.tensor(logits_cal), dim=1).numpy()
        labels_hard_cal = hard_labels_all[idx_cal]
        labels_soft_cal = soft_labels_all[idx_cal]
        labels_hard_te = hard_labels_all[idx_te]
        labels_soft_te = soft_labels_all[idx_te]

        per_seed[seed] = {}
        for method in METHOD_ORDER:
            metrics = fit_and_eval_method(
                method=method,
                logits_cal=logits_cal,
                probs_cal=probs_cal,
                labels_hard_cal=labels_hard_cal,
                labels_soft_cal=labels_soft_cal,
                logits_te=logits_te,
                labels_hard_te=labels_hard_te,
                labels_soft_te=labels_soft_te,
                n_bins=args.n_bins,
                adam_epochs=args.adam_epochs,
                device=device,
            )
            per_seed[seed][method] = metrics["ece_sampled"]
            print(f"  {method:<15} ECE_true={metrics['ece_sampled']*100:5.2f}%")

    # ── Summary statistics ────────────────────────────────────────────────────
    summary = {}
    for method in METHOD_ORDER:
        vals = [per_seed[s][method] for s in args.seeds]
        summary[method] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_seed": vals,
        }

    print("\n" + "="*50)
    print(f"  {'Method':<18} {'Mean ECE_true':>14} {'Std':>7}")
    print("  " + "-"*40)
    for method in METHOD_ORDER:
        m, s = summary[method]["mean"], summary[method]["std"]
        print(f"  {method:<18} {m*100:>13.2f}% {s*100:>6.2f}%")
    print("="*50)

    results = {
        "dataset": args.dataset,
        "arch": args.arch,
        "seeds": args.seeds,
        "per_seed_results": {str(s): per_seed[s] for s in args.seeds},
        "summary": summary,
    }

    suffix = f"_{args.tag}" if args.tag else ""
    out_json = results_dir / f"multiseed_{args.dataset}_{args.arch}{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {out_json}")

    out_pdf = figures_dir / f"multiseed_{args.dataset}_{args.arch}{suffix}.pdf"
    out_png = figures_dir / f"multiseed_{args.dataset}_{args.arch}{suffix}.png"
    make_plot(results, out_pdf, out_png)
    print(f"Saved plot → {out_pdf}")


if __name__ == "__main__":
    main()
