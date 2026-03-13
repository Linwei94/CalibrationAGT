"""
Empirical validation of Proposition 2 (entropy-gap relationship).

For CIFAR-10H (ResNet-50 and ViT-B/16) and ChaosNLI (DeBERTa-v3):
- Bin test examples by annotation entropy H(x)
- Compute per-bin ECE_true for TS and SLTS (voted-label vs ambiguity-aware)
- Show that the true-label calibration error of TS increases with H(x)

Usage
-----
    cd experiments/
    python plot_entropy_validation.py [--cache-dir ./cache] [--results-dir ./results]
                                       [--figures-dir ./figures] [--seed 42] [--n-bins 5]
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from calibration import TemperatureScaling, SoftLabelTS, apply_parametric


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (same split logic as run_lsts_ablation.py)
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
    """
    Per-example true-label calibration error.
    Uses sampled-label ECE: for each example, draw a label from soft_labels
    and compute expected |confidence - accuracy|.

    Here we use the expected value: E_y~pi[|hat_p(y_hat) - 1(y=y_hat)|].
    Simplified: ECE contribution = |p_top - pi_top| where p_top = max(probs)
    and pi_top = soft_labels for the predicted class.
    """
    pred_class = probs.argmax(axis=1)
    p_top = probs[np.arange(len(probs)), pred_class]
    pi_top = soft_labels[np.arange(len(soft_labels)), pred_class]
    return np.abs(p_top - pi_top)


def compute_entropy_bins(probs_ts, probs_slts, soft_te, n_entropy_bins=5):
    """
    Bin test examples by annotation entropy, compute mean per-example
    true-label calibration error in each bin.
    """
    H = annotation_entropy(soft_te)
    bin_edges = np.percentile(H, np.linspace(0, 100, n_entropy_bins + 1))
    bin_edges[-1] += 1e-6  # include max

    err_ts   = ece_true_per_example(probs_ts, soft_te)
    err_slts = ece_true_per_example(probs_slts, soft_te)

    H_centers, ts_mean, slts_mean, ts_std, slts_std, counts = [], [], [], [], [], []
    for b in range(n_entropy_bins):
        mask = (H >= bin_edges[b]) & (H < bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        H_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        ts_mean.append(err_ts[mask].mean())
        slts_mean.append(err_slts[mask].mean())
        ts_std.append(err_ts[mask].std() / np.sqrt(mask.sum()))
        slts_std.append(err_slts[mask].std() / np.sqrt(mask.sum()))
        counts.append(mask.sum())

    return (np.array(H_centers), np.array(ts_mean), np.array(slts_mean),
            np.array(ts_std), np.array(slts_std), np.array(counts))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "TS":   {"color": "#d62728", "marker": "o", "label": "TS (voted-label)"},
    "SLTS": {"color": "#1f77b4", "marker": "s", "label": "SLTS (soft-label)"},
}


def plot_panel(ax, H_centers, ts_mean, slts_mean, ts_std, slts_std, title, n_entropy_bins):
    x = np.arange(len(H_centers))
    ax.errorbar(x, ts_mean * 100, yerr=ts_std * 100,
                color=STYLE["TS"]["color"], marker=STYLE["TS"]["marker"],
                linewidth=1.5, markersize=5, capsize=3, label="TS")
    ax.errorbar(x, slts_mean * 100, yerr=slts_std * 100,
                color=STYLE["SLTS"]["color"], marker=STYLE["SLTS"]["marker"],
                linewidth=1.5, markersize=5, capsize=3, label="SLTS")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in H_centers], fontsize=8)
    ax.set_xlabel("Annotation entropy $H(x)/\\log K$", fontsize=9)
    ax.set_ylabel("Pointwise calibration error (%)", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir",   default="./cache")
    parser.add_argument("--figures-dir", default="./figures")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n-entropy-bins", type=int, default=5)
    args = parser.parse_args()

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    configs = [
        ("CIFAR-10H\nResNet-50",  load_cifar10h, "resnet50"),
        ("CIFAR-10H\nViT-B/16",   load_cifar10h, "vit_b16"),
        ("ChaosNLI\nDeBERTa-v3",  load_chaosnli, "deberta_v3"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    for ax, (title, loader, arch) in zip(axes, configs):
        print(f"[{title.replace(chr(10), ' ')}] Loading…")
        lc, lt, yh_c, yh_t, ys_c, ys_t = loader(args.cache_dir, arch, args.seed)

        yh_c_t = torch.tensor(yh_c, dtype=torch.long)
        ys_c_t = torch.tensor(ys_c, dtype=torch.float32)

        # Fit TS and SLTS
        ts   = TemperatureScaling().fit(lc, yh_c_t)
        slts = SoftLabelTS().fit(lc, ys_c_t)

        probs_ts   = apply_parametric(ts,   lt.numpy())
        probs_slts = apply_parametric(slts, lt.numpy())

        H_c, ts_m, slts_m, ts_s, slts_s, cnts = compute_entropy_bins(
            probs_ts, probs_slts, ys_t, n_entropy_bins=args.n_entropy_bins
        )
        print(f"  Bins: {cnts} | H_centers: {H_c.round(3)}")
        print(f"  TS    ECE%: {np.round(ts_m*100, 2)}")
        print(f"  SLTS  ECE%: {np.round(slts_m*100, 2)}")

        plot_panel(ax, H_c, ts_m, slts_m, ts_s, slts_s, title, args.n_entropy_bins)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color=STYLE["TS"]["color"],   marker="o", linewidth=1.5, markersize=5, label="TS (voted-label)"),
        Line2D([0], [0], color=STYLE["SLTS"]["color"], marker="s", linewidth=1.5, markersize=5, label="SLTS (soft-label)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()

    out_pdf = Path(args.figures_dir) / "fig_entropy_validation.pdf"
    out_png = Path(args.figures_dir) / "fig_entropy_validation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"\nSaved → {out_pdf}")


if __name__ == "__main__":
    main()
