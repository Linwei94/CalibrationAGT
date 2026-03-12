"""
Per-class ε ablation: compare LS-TS (global ε) vs CC-LS-TS (class-conditional ε).

For each dataset/arch, fits both methods and compares ECE_true, Brier, NLL.
Also reports per-class ε values to show which classes are most ambiguous.

Datasets: cifar10h, chaosnli, isic2019, dermamnist

Usage
-----
    python run_lsts_perclass.py --dataset cifar10h --arch resnet50
    python run_lsts_perclass.py --dataset chaosnli --arch roberta_large
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
    LabelSmoothTS, ClassCondLabelSmoothTS, TemperatureScaling,
    SoftLabelTS, apply_parametric,
)
from metrics import compute_all_metrics
from run_cal_size_ablation import load_cifar10h_data, load_chaosnli_data


def load_isic_data(cache_dir: Path, arch: str, seed: int):
    """Load ISIC 2019 data (mirrors run_isic2019.py split logic)."""
    from run_isic2019 import load_isic2019_data
    return load_isic2019_data(str(cache_dir), arch, seed)


def load_derm_data(cache_dir: Path, arch: str, seed: int):
    """Load DermaMNIST data."""
    from run_dermamnist import load_dermamnist_data
    return load_dermamnist_data(str(cache_dir), arch, seed)


CLASS_NAMES = {
    "cifar10h": ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"],
    "chaosnli": ["entailment", "neutral", "contradiction"],
    "isic2019": ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"],
    "dermamnist": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
}


def run_perclass(
    logits_cal, labels_hard_cal, labels_soft_cal,
    logits_te, labels_hard_te, labels_soft_te,
    n_bins=15,
):
    logits_cal_t = torch.tensor(logits_cal, dtype=torch.float32)
    labels_hard_cal_t = torch.tensor(labels_hard_cal, dtype=torch.long)
    labels_soft_cal_t = torch.tensor(labels_soft_cal, dtype=torch.float32)

    # TS baseline
    ts = TemperatureScaling().fit(logits_cal_t, labels_hard_cal_t)
    p_ts = apply_parametric(ts, logits_te)
    m_ts = compute_all_metrics(p_ts, labels_hard_te, labels_soft_te,
                               n_bins=n_bins, name="TS")
    m_ts["T"] = ts.T

    # LS-TS (global ε)
    ls = LabelSmoothTS().fit(logits_cal_t, labels_hard_cal_t)
    p_ls = apply_parametric(ls, logits_te)
    m_ls = compute_all_metrics(p_ls, labels_hard_te, labels_soft_te,
                               n_bins=n_bins, name="LS-TS")
    # Compute global ε
    with torch.no_grad():
        f = torch.softmax(logits_cal_t, dim=1)
        conf = f[torch.arange(len(labels_hard_cal_t)), labels_hard_cal_t]
        eps_global = (1.0 - conf).mean().item()
    m_ls["T"] = ls.T
    m_ls["eps_global"] = eps_global

    # CC-LS-TS (per-class ε)
    cc = ClassCondLabelSmoothTS().fit(logits_cal_t, labels_hard_cal_t)
    p_cc = apply_parametric(cc, logits_te)
    m_cc = compute_all_metrics(p_cc, labels_hard_te, labels_soft_te,
                               n_bins=n_bins, name="CC-LS-TS")
    m_cc["T"] = cc.T
    m_cc["eps_per_class"] = cc.eps_per_class

    # SLTS (oracle, for reference)
    slts = SoftLabelTS().fit(logits_cal_t, labels_soft_cal_t)
    p_slts = apply_parametric(slts, logits_te)
    m_slts = compute_all_metrics(p_slts, labels_hard_te, labels_soft_te,
                                 n_bins=n_bins, name="SLTS")
    m_slts["T"] = slts.T

    # Oracle per-class ε from annotator distributions
    K = labels_soft_cal.shape[1]
    eps_oracle_per_class = []
    for k in range(K):
        mask = labels_hard_cal == k
        if mask.sum() > 0:
            pi_voted = labels_soft_cal[mask, k]
            eps_oracle_per_class.append(float(1.0 - pi_voted.mean()))
        else:
            eps_oracle_per_class.append(0.0)

    return m_ts, m_ls, m_cc, m_slts, eps_oracle_per_class


def make_plot(m_cc, eps_oracle, class_names, dataset, arch, out_pdf, out_png):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    K = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, K * 0.9), 4.5))

    x = np.arange(K)
    width = 0.35
    bars1 = ax.bar(x - width/2, m_cc["eps_per_class"], width,
                   label="CC-LS-TS (estimated)", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width/2, eps_oracle, width,
                   label="Oracle (from annotators)", color="#4CAF50", alpha=0.8)

    ax.set_xlabel("Class")
    ax.set_ylabel("ε (smoothing)")
    ax.set_title(f"Per-class ε — {dataset} ({arch})")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45 if K > 5 else 0, ha="right" if K > 5 else "center")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10h", "chaosnli", "isic2019", "dermamnist"],
                   default="cifar10h")
    p.add_argument("--arch", default="resnet50")
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
    elif args.dataset == "chaosnli":
        logits_all, hard_labels_all, soft_labels_all, idx_cal, idx_te, rng = \
            load_chaosnli_data(cache_dir, args.arch, args.seed)
    elif args.dataset == "isic2019":
        logits_all, hard_labels_all, soft_labels_all, idx_cal, idx_te, rng = \
            load_isic_data(cache_dir, args.arch, args.seed)
    else:
        logits_all, hard_labels_all, soft_labels_all, idx_cal, idx_te, rng = \
            load_derm_data(cache_dir, args.arch, args.seed)

    print(f"Dataset: {args.dataset} ({args.arch})")
    print(f"Cal n={len(idx_cal)}, Test n={len(idx_te)}\n")

    m_ts, m_ls, m_cc, m_slts, eps_oracle = run_perclass(
        logits_all[idx_cal], hard_labels_all[idx_cal], soft_labels_all[idx_cal],
        logits_all[idx_te], hard_labels_all[idx_te], soft_labels_all[idx_te],
        n_bins=args.n_bins,
    )

    class_names = CLASS_NAMES.get(args.dataset, [f"C{k}" for k in range(len(eps_oracle))])

    print("=" * 70)
    print(f"{'Method':<15} {'T':>6} {'ECE_true':>10} {'Brier':>8} {'NLL':>8}")
    print("-" * 70)
    for name, m in [("TS", m_ts), ("LS-TS", m_ls), ("CC-LS-TS", m_cc), ("SLTS", m_slts)]:
        print(f"{name:<15} {m['T']:6.3f} {m['ece_sampled']*100:9.2f}% "
              f"{m['brier_sampled']:8.4f} {m['nll_sampled']:8.4f}")

    print(f"\nGlobal ε (LS-TS): {m_ls['eps_global']:.4f}")
    print(f"\nPer-class ε (CC-LS-TS vs Oracle):")
    for k, cn in enumerate(class_names):
        est = m_cc["eps_per_class"][k] if k < len(m_cc["eps_per_class"]) else 0
        orc = eps_oracle[k] if k < len(eps_oracle) else 0
        print(f"  {cn:<15} estimated={est:.4f}  oracle={orc:.4f}  "
              f"Δ={est-orc:+.4f}")

    out = {
        "dataset": args.dataset, "arch": args.arch, "seed": args.seed,
        "class_names": class_names,
        "ts": m_ts, "lsts": m_ls, "cc_lsts": m_cc, "slts": m_slts,
        "eps_oracle_per_class": eps_oracle,
    }
    out_json = results_dir / f"perclass_eps_{args.dataset}_{args.arch}.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {out_json}")

    out_pdf = figures_dir / f"perclass_eps_{args.dataset}_{args.arch}.pdf"
    out_png = figures_dir / f"perclass_eps_{args.dataset}_{args.arch}.png"
    make_plot(m_cc, eps_oracle, class_names, args.dataset, args.arch, out_pdf, out_png)
    print(f"Plot  → {out_pdf}")


if __name__ == "__main__":
    main()
