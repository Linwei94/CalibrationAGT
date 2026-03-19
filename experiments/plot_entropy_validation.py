"""
Empirical validation of Proposition 2 (entropy-gap relationship).

For CIFAR-10H (ResNet-50 and ViT-B/16) and ChaosNLI (DeBERTa-v3):
- Bin test examples by annotation entropy H(x)
- Compute per-bin pointwise true-label calibration error for TS
- Show that the error increases with H(x), confirming Proposition 2
- All three dataset/arch curves on a single-column figure

Usage
-----
    cd experiments/
    python plot_entropy_validation.py [--cache-dir ./cache] [--figures-dir ../paper/figs]
                                       [--seed 42] [--n-entropy-bins 5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from calibration import TemperatureScaling, apply_parametric


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (same split logic as main experiments)
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10h(cache_dir: str, arch: str, seed: int):
    logits_all = np.load(Path(cache_dir) / f"logits_test_{arch}.npy")
    soft_all   = np.load(Path(cache_dir) / "cifar10h-probs.npy").astype(np.float32)
    hard_all   = soft_all.argmax(axis=1)
    N, K       = logits_all.shape

    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(K):
        c_idx  = idx[hard_all == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True

    ic, it = idx[cal_mask], idx[~cal_mask]
    return (
        torch.tensor(logits_all[ic], dtype=torch.float32),
        torch.tensor(logits_all[it], dtype=torch.float32),
        hard_all[ic], hard_all[it],
        soft_all[ic], soft_all[it],
    )


def load_chaosnli(cache_dir: str, arch: str, seed: int):
    import json as _json
    data_dir = Path(cache_dir) / "chaosNLI_v1.0"
    raw = []
    for fname in ("chaosNLI_snli.jsonl", "chaosNLI_mnli_m.jsonl"):
        with open(data_dir / fname) as f:
            for line in f:
                raw.append(_json.loads(line.strip()))

    soft_list, hard_list = [], []
    for d in raw:
        ld = d.get("label_dist")
        if isinstance(ld, list):
            soft = np.array(ld, dtype=np.float32)
        else:
            counter = d.get("label_counter", {})
            counts  = np.array([
                counter.get("e", counter.get("entailment", 0)),
                counter.get("n", counter.get("neutral", 0)),
                counter.get("c", counter.get("contradiction", 0)),
            ], dtype=np.float32)
            total = counts.sum()
            soft = counts / total if total > 0 else np.ones(3) / 3
        soft_list.append(soft)
        hard_list.append(int(soft.argmax()))
    soft_all = np.array(soft_list, dtype=np.float32)
    hard_all = np.array(hard_list, dtype=np.int64)

    logits_all = np.load(Path(cache_dir) / f"logits_chaosnli_combined_{arch}.npy")
    N, K = logits_all.shape
    rng  = np.random.default_rng(seed)
    idx  = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(K):
        c_idx  = idx[hard_all == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True

    ic, it = idx[cal_mask], idx[~cal_mask]
    return (
        torch.tensor(logits_all[ic], dtype=torch.float32),
        torch.tensor(logits_all[it], dtype=torch.float32),
        hard_all[ic], hard_all[it],
        soft_all[ic], soft_all[it],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-bin ECE computation
# ─────────────────────────────────────────────────────────────────────────────

def annotation_entropy(soft: np.ndarray) -> np.ndarray:
    """Normalized annotation entropy H(x)/log(K) for each example."""
    K = soft.shape[1]
    eps = 1e-12
    h = -np.sum(soft * np.log(soft + eps), axis=1)
    log_k = np.log(K)
    return h / log_k


def ece_true_per_example(probs: np.ndarray, soft_labels: np.ndarray) -> np.ndarray:
    """Per-example pointwise true-label calibration error: |p_top - pi_top|."""
    pred_class = probs.argmax(axis=1)
    p_top = probs[np.arange(len(probs)), pred_class]
    pi_top = soft_labels[np.arange(len(soft_labels)), pred_class]
    return np.abs(p_top - pi_top)


def compute_entropy_bins_ts(probs_ts, soft_te, n_entropy_bins=5):
    """Bin test examples by annotation entropy, return per-bin TS error stats."""
    H = annotation_entropy(soft_te)
    bin_edges = np.percentile(H, np.linspace(0, 100, n_entropy_bins + 1))
    bin_edges[-1] += 1e-6

    err_ts = ece_true_per_example(probs_ts, soft_te)

    H_centers, ts_mean, ts_se = [], [], []
    for b in range(n_entropy_bins):
        mask = (H >= bin_edges[b]) & (H < bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        H_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        ts_mean.append(err_ts[mask].mean())
        ts_se.append(err_ts[mask].std() / np.sqrt(mask.sum()))

    return np.array(H_centers), np.array(ts_mean), np.array(ts_se)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting style
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS_STYLE = {
    "CIFAR-10H ResNet-50":  {"color": "#d62728", "marker": "o"},
    "CIFAR-10H ViT-B/16":   {"color": "#ff7f0e", "marker": "s"},
    "ChaosNLI DeBERTa-v3":  {"color": "#1f77b4", "marker": "D"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir",      default="./cache")
    parser.add_argument("--figures-dir",     default="../paper/figs")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--n-entropy-bins",  type=int, default=5)
    args = parser.parse_args()

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    configs = [
        ("CIFAR-10H ResNet-50",  load_cifar10h, "resnet50"),
        ("CIFAR-10H ViT-B/16",   load_cifar10h, "vit_b16"),
        ("ChaosNLI DeBERTa-v3",  load_chaosnli, "deberta_v3"),
    ]

    # Single-column figure (IEEEtran single-column width ~3.5in)
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

    for label, loader, arch in configs:
        print(f"[{label}] Loading...")
        lc, lt, yh_c, yh_t, ys_c, ys_t = loader(args.cache_dir, arch, args.seed)

        yh_c_t = torch.tensor(yh_c, dtype=torch.long)

        # Fit TS only
        ts = TemperatureScaling().fit(lc, yh_c_t)
        probs_ts = apply_parametric(ts, lt.numpy())

        H_c, ts_m, ts_se = compute_entropy_bins_ts(
            probs_ts, ys_t, n_entropy_bins=args.n_entropy_bins
        )
        print(f"  H_centers: {H_c.round(3)}")
        print(f"  TS ECE%:   {np.round(ts_m * 100, 2)}")

        style = CONFIGS_STYLE[label]
        ax.errorbar(
            H_c, ts_m * 100, yerr=ts_se * 100,
            color=style["color"], marker=style["marker"],
            linewidth=1.4, markersize=5, capsize=3,
            label=label,
        )

    ax.set_xlabel(r"Normalised annotation entropy $H(x)/\log K$", fontsize=9)
    ax.set_ylabel("Pointwise calibration error (%)", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, fancybox=False,
              edgecolor="0.7")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)

    fig.tight_layout()

    out_pdf = Path(args.figures_dir) / "fig_entropy_validation.pdf"
    out_png = Path(args.figures_dir) / "fig_entropy_validation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"\nSaved -> {out_pdf}")


if __name__ == "__main__":
    main()
