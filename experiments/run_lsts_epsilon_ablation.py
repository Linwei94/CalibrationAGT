"""
Epsilon sensitivity ablation for LS-TS.

Fixes ε at different values (0.02 .. 0.50), optimises T for each,
and compares against:
  - LS-TS's self-estimated ε̂
  - Oracle ε* = mean(1 − π_{y*}) from real annotator distributions

This shows (a) how sensitive ECE_true is to ε, and (b) how close the
self-estimated ε̂ is to the oracle.

Datasets: cifar10h, chaosnli

Usage
-----
    python run_lsts_epsilon_ablation.py --dataset cifar10h --arch resnet50
    python run_lsts_epsilon_ablation.py --dataset chaosnli --arch roberta_large
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
from calibration import LabelSmoothTS, apply_parametric
from metrics import compute_all_metrics
from run_cal_size_ablation import load_cifar10h_data, load_chaosnli_data

DEFAULT_EPSILONS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


class FixedEpsilonTS(torch.nn.Module):
    """TS with a fixed smoothing parameter ε."""

    def __init__(self, eps: float, init_T: float = 1.5):
        super().__init__()
        self.eps = eps
        self.temperature = torch.nn.Parameter(torch.ones(1) * init_T)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits, labels_hard):
        K = logits.shape[1]
        with torch.no_grad():
            yh = torch.zeros(len(labels_hard), K, dtype=torch.float32)
            yh.scatter_(1, labels_hard.unsqueeze(1), 1.0)
            pi_hat = (1 - self.eps) * yh + (self.eps / K)

        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                  tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            opt.zero_grad()
            log_p = torch.log_softmax(self(logits), dim=1)
            loss = -(pi_hat * log_p).sum(1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @property
    def T(self):
        return self.temperature.item()


def compute_oracle_epsilon(labels_soft: np.ndarray, labels_hard: np.ndarray) -> float:
    """Oracle ε* = mean(1 - π_{y*}) from real annotator distributions."""
    pi_voted = labels_soft[np.arange(len(labels_hard)), labels_hard]
    return float(1.0 - pi_voted.mean())


def compute_lsts_epsilon(logits: np.ndarray, labels_hard: np.ndarray) -> float:
    """LS-TS's self-estimated ε̂ = mean(1 - f_{y*})."""
    with torch.no_grad():
        f = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)
        conf = f[torch.arange(len(labels_hard)),
                 torch.tensor(labels_hard, dtype=torch.long)]
    return float((1.0 - conf).mean().item())


def run_epsilon_ablation(
    logits_cal, labels_hard_cal, labels_soft_cal,
    logits_te, labels_hard_te, labels_soft_te,
    eps_list, n_bins=15,
):
    logits_cal_t = torch.tensor(logits_cal, dtype=torch.float32)
    labels_hard_cal_t = torch.tensor(labels_hard_cal, dtype=torch.long)

    results = []
    for eps in eps_list:
        cal = FixedEpsilonTS(eps).fit(logits_cal_t, labels_hard_cal_t)
        probs = apply_parametric(cal, logits_te)
        m = compute_all_metrics(probs, labels_hard_te, labels_soft_te,
                                n_bins=n_bins, name=f"ε={eps:.2f}")
        m["eps"] = eps
        m["T"] = cal.T
        m["ece_true"] = m["ece_sampled"]
        results.append(m)
        print(f"  ε={eps:.2f}  T={cal.T:.4f}  ECE_true={m['ece_sampled']*100:.2f}%  "
              f"Br={m['brier_sampled']:.4f}  NLL={m['nll_sampled']:.4f}")

    # Also run standard LS-TS
    ls = LabelSmoothTS().fit(logits_cal_t, labels_hard_cal_t)
    probs_ls = apply_parametric(ls, logits_te)
    m_ls = compute_all_metrics(probs_ls, labels_hard_te, labels_soft_te,
                               n_bins=n_bins, name="LS-TS")
    eps_hat = compute_lsts_epsilon(logits_cal, labels_hard_cal)
    m_ls["eps"] = eps_hat
    m_ls["T"] = ls.T
    m_ls["ece_true"] = m_ls["ece_sampled"]

    eps_oracle = compute_oracle_epsilon(labels_soft_cal, labels_hard_cal)

    return results, m_ls, eps_hat, eps_oracle


def make_plot(results, m_ls, eps_hat, eps_oracle, dataset, arch, out_pdf, out_png):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    eps_vals = [r["eps"] for r in results]
    ece_vals = [r["ece_true"] * 100 for r in results]
    t_vals = [r["T"] for r in results]

    # Panel 1: ECE_true vs ε
    ax = axes[0]
    ax.plot(eps_vals, ece_vals, "o-", color="#2196F3", lw=2, label="Fixed ε")
    ax.axvline(eps_hat, color="#E91E63", ls="--", lw=1.5, label=f"LS-TS ε̂={eps_hat:.3f}")
    ax.axvline(eps_oracle, color="#4CAF50", ls="--", lw=1.5, label=f"Oracle ε*={eps_oracle:.3f}")
    ax.plot(eps_hat, m_ls["ece_true"] * 100, "*", color="#E91E63", ms=14, zorder=5)
    ax.set_xlabel("Smoothing ε")
    ax.set_ylabel("ECE_true (%)")
    ax.set_title("(a) ECE_true vs ε")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 2: T vs ε
    ax = axes[1]
    ax.plot(eps_vals, t_vals, "s-", color="#FF9800", lw=2, label="Fixed ε")
    ax.axvline(eps_hat, color="#E91E63", ls="--", lw=1.5, label=f"LS-TS ε̂={eps_hat:.3f}")
    ax.axvline(eps_oracle, color="#4CAF50", ls="--", lw=1.5, label=f"Oracle ε*={eps_oracle:.3f}")
    ax.plot(eps_hat, m_ls["T"], "*", color="#E91E63", ms=14, zorder=5)
    ax.set_xlabel("Smoothing ε")
    ax.set_ylabel("Temperature T")
    ax.set_title("(b) Learned T vs ε")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"ε sensitivity — {dataset} ({arch})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10h", "chaosnli"], default="cifar10h")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
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
    print(f"Epsilons: {args.epsilons}\n")

    results, m_ls, eps_hat, eps_oracle = run_epsilon_ablation(
        logits_cal=logits_all[idx_cal],
        labels_hard_cal=hard_labels_all[idx_cal],
        labels_soft_cal=soft_labels_all[idx_cal],
        logits_te=logits_all[idx_te],
        labels_hard_te=hard_labels_all[idx_te],
        labels_soft_te=soft_labels_all[idx_te],
        eps_list=sorted(args.epsilons),
        n_bins=args.n_bins,
    )

    print(f"\n  LS-TS auto ε̂ = {eps_hat:.4f}")
    print(f"  Oracle   ε* = {eps_oracle:.4f}")
    print(f"  LS-TS ECE_true = {m_ls['ece_true']*100:.2f}%")

    out = {
        "dataset": args.dataset, "arch": args.arch, "seed": args.seed,
        "eps_hat": eps_hat, "eps_oracle": eps_oracle,
        "lsts_metrics": m_ls,
        "fixed_eps_results": results,
    }
    out_json = results_dir / f"eps_ablation_{args.dataset}_{args.arch}.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {out_json}")

    out_pdf = figures_dir / f"eps_ablation_{args.dataset}_{args.arch}.pdf"
    out_png = figures_dir / f"eps_ablation_{args.dataset}_{args.arch}.png"
    make_plot(results, m_ls, eps_hat, eps_oracle,
              args.dataset, args.arch, out_pdf, out_png)
    print(f"Plot  → {out_pdf}")


if __name__ == "__main__":
    main()
