"""
EM iteration ablation for LS-TS.

Tests whether iterating the E-step (re-estimate ε from calibrated softmax)
and M-step (re-optimise T) improves over the single-round LS-TS.

Datasets: cifar10h, chaosnli
Rounds: 1 (=LS-TS), 2, 3, 5, 10

Usage
-----
    python run_lsts_em_ablation.py --dataset cifar10h --arch resnet50
    python run_lsts_em_ablation.py --dataset chaosnli --arch roberta_large
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
from calibration import EMSmoothTS, LabelSmoothTS, apply_parametric
from metrics import compute_all_metrics
from run_cal_size_ablation import load_cifar10h_data, load_chaosnli_data

DEFAULT_ROUNDS = [1, 2, 3, 5, 10]


def run_em_ablation(
    logits_cal: np.ndarray,
    labels_hard_cal: np.ndarray,
    logits_te: np.ndarray,
    labels_hard_te: np.ndarray,
    labels_soft_te: np.ndarray,
    n_rounds_list: list[int],
    n_bins: int = 15,
) -> list[dict]:
    results = []
    labels_hard_cal_t = torch.tensor(labels_hard_cal, dtype=torch.long)
    logits_cal_t = torch.tensor(logits_cal, dtype=torch.float32)

    for n_rounds in n_rounds_list:
        cal = EMSmoothTS().fit(logits_cal_t, labels_hard_cal_t, n_rounds=n_rounds)
        probs = apply_parametric(cal, logits_te)
        metrics = compute_all_metrics(probs, labels_hard_te, labels_soft_te,
                                      n_bins=n_bins, name=f"EM-LS-TS (R={n_rounds})")
        metrics["n_rounds"] = n_rounds
        metrics["T"] = cal.T
        metrics["history"] = cal.history
        metrics["ece_true"] = metrics["ece_sampled"]
        results.append(metrics)
        print(f"  R={n_rounds:2d}  T={cal.T:.4f}  ε={cal.history[-1]['eps']:.4f}  "
              f"ECE_true={metrics['ece_sampled']*100:.2f}%")

    return results


def make_plot(results: list[dict], dataset: str, arch: str,
              out_pdf: Path, out_png: Path):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    rounds = [r["n_rounds"] for r in results]
    ece = [r["ece_true"] * 100 for r in results]
    temps = [r["T"] for r in results]
    epsilons = [r["history"][-1]["eps"] for r in results]

    # Panel 1: ECE_true vs rounds
    axes[0].plot(rounds, ece, "o-", color="#2196F3", lw=2)
    axes[0].set_xlabel("EM rounds")
    axes[0].set_ylabel("ECE_true (%)")
    axes[0].set_title("(a) Calibration quality")
    axes[0].set_xticks(rounds)
    axes[0].grid(True, alpha=0.25)

    # Panel 2: T vs rounds
    axes[1].plot(rounds, temps, "s-", color="#4CAF80", lw=2)
    axes[1].set_xlabel("EM rounds")
    axes[1].set_ylabel("Temperature T")
    axes[1].set_title("(b) Learned temperature")
    axes[1].set_xticks(rounds)
    axes[1].grid(True, alpha=0.25)

    # Panel 3: ε vs rounds
    axes[2].plot(rounds, epsilons, "^-", color="#E91E63", lw=2)
    axes[2].set_xlabel("EM rounds")
    axes[2].set_ylabel("Smoothing ε")
    axes[2].set_title("(c) Estimated ε")
    axes[2].set_xticks(rounds)
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"EM-LS-TS ablation — {dataset} ({arch})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10h", "chaosnli"], default="cifar10h")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--rounds", nargs="+", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--cache-dir", default="experiments/cache")
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--figures-dir", default="experiments/figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bins", type=int, default=15)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10h":
        logits_all, hard_labels_all, soft_labels_all, idx_cal, idx_te, rng = \
            load_cifar10h_data(cache_dir, args.arch, args.seed)
    else:
        logits_all, hard_labels_all, soft_labels_all, idx_cal, idx_te, rng = \
            load_chaosnli_data(cache_dir, args.arch, args.seed)

    print(f"Dataset: {args.dataset} ({args.arch})")
    print(f"Cal n={len(idx_cal)}, Test n={len(idx_te)}")
    print(f"Rounds: {args.rounds}\n")

    results = run_em_ablation(
        logits_cal=logits_all[idx_cal],
        labels_hard_cal=hard_labels_all[idx_cal],
        logits_te=logits_all[idx_te],
        labels_hard_te=hard_labels_all[idx_te],
        labels_soft_te=soft_labels_all[idx_te],
        n_rounds_list=sorted(args.rounds),
        n_bins=args.n_bins,
    )

    out = {
        "dataset": args.dataset, "arch": args.arch, "seed": args.seed,
        "results": results,
    }
    out_json = results_dir / f"em_ablation_{args.dataset}_{args.arch}.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {out_json}")

    out_pdf = figures_dir / f"em_ablation_{args.dataset}_{args.arch}.pdf"
    out_png = figures_dir / f"em_ablation_{args.dataset}_{args.arch}.png"
    make_plot(results, args.dataset, args.arch, out_pdf, out_png)
    print(f"Plot  → {out_pdf}")


if __name__ == "__main__":
    main()
