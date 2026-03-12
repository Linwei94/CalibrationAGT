"""
Regenerate fig4_stratified.pdf — ECE-Soft by annotation entropy quartile (CIFAR-10H).

Uses rank-based quartile splitting so Q1 is never empty even when many
samples share entropy = 0 (unanimous votes).

Line plot instead of bar chart, showing both ResNet-50 and ViT-B/16.
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CACHE      = ROOT / "experiments" / "cache"
RESULTS    = ROOT / "experiments" / "results"
OUT        = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
PAL = dict(
    ts    ="#E74C3C",
    ps    ="#E67E22",
    slts  ="#2980B9",
    mcts  ="#27AE60",
    lsts  ="#8E44AD",
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def annotation_entropy(ys, eps=1e-12):
    ys = np.clip(ys, eps, 1.0)
    return -np.sum(ys * np.log(ys), axis=1)

def compute_ece(probs, labels_soft, n_bins=15):
    """ECE-Soft: labels_soft can be soft (K-dim) or one-hot."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        acc = labels_soft[m][np.arange(m.sum()), pred[m]].mean()
        ece += m.sum() / n * abs(conf[m].mean() - acc)
    return ece

def apply_temperature(logits_np, T):
    """Scale logits by temperature and return softmax probabilities."""
    return torch.softmax(torch.tensor(logits_np / T), dim=1).numpy()

def rank_quartile_ece(probs, ys_te, n_bins=15):
    """
    Split test set into 4 equal-size groups by annotation entropy rank.
    Returns list of 4 ECE values (one per quartile, Q1=lowest entropy).
    """
    H = annotation_entropy(ys_te)
    sorted_idx = np.argsort(H, kind="stable")
    n = len(H)
    ece_per_q = []
    for q in range(4):
        lo = q * n // 4
        hi = (q + 1) * n // 4
        idx_q = sorted_idx[lo:hi]
        ece_q = compute_ece(probs[idx_q], ys_te[idx_q], n_bins=n_bins)
        ece_per_q.append(ece_q * 100)
    return ece_per_q


def load_cifar10h():
    """Return soft labels array (10000, 10)."""
    path = CACHE / "cifar10h-probs.npy"
    return np.load(path)


def get_test_idx(hard_labels, seed=42):
    """Reproduce the 50/50 stratified split used in run_cifar10h.py."""
    rng = np.random.default_rng(seed)
    idx = np.arange(10000)
    cal_mask = np.zeros(10000, dtype=bool)
    n_classes = 10
    for c in range(n_classes):
        c_idx  = idx[hard_labels == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True
    idx_te = idx[~cal_mask]
    return idx_te


# ── Load CIFAR-10 hard labels ──────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / "experiments"))
from run_cifar10h import get_cifar10_testset

testset    = get_cifar10_testset(str(CACHE / "cifar10"))
hard_labels = np.array([y for _, y in testset])   # (10000,)

cifar10h  = load_cifar10h()                        # (10000, 10)
idx_te    = get_test_idx(hard_labels, seed=42)

ys_te     = cifar10h[idx_te]                       # (5000, 10)
H_te      = annotation_entropy(ys_te)

print(f"Test set entropy stats:  min={H_te.min():.4f}  median={np.median(H_te):.4f}  max={H_te.max():.4f}")
print(f"Samples with H=0: {(H_te == 0).sum()} / {len(H_te)}")

# ── Per-arch processing ────────────────────────────────────────────────────────
archs = [
    ("ResNet-50", "resnet50",  "cifar10h_results_resnet50.json"),
    ("ViT-B/16",  "vit_b16",   "cifar10h_results_vit_b16.json"),
]

# methods: (label, color, json_key_for_T, use_ts_T_for_platt)
# We'll compute ECE for TS, SLTS, and LS-TS
METHODS = [
    ("TS",       PAL["ts"],   "ts_temperature"),
    ("SLTS",     PAL["slts"], "slts_temperature"),
    ("LS-TS",    PAL["lsts"], "ls_ts_temperature"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
x = np.array([1, 2, 3, 4])
x_labels = ["Q1\n(unanimous)", "Q2", "Q3", "Q4\n(ambiguous)"]

for ax, (arch_label, arch_key, json_file) in zip(axes, archs):
    res_path = RESULTS / json_file
    with open(res_path) as f:
        res = json.load(f)

    logits_all = np.load(CACHE / f"logits_test_{arch_key}.npy")  # (10000, 10)
    logits_te  = logits_all[idx_te]                               # (5000, 10)

    for method_label, col, T_key in METHODS:
        T = res[T_key]
        probs = apply_temperature(logits_te, T)
        ece_q = rank_quartile_ece(probs, ys_te)
        ax.plot(x, ece_q, marker="o", color=col, label=method_label,
                linewidth=2.0, markersize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Annotation entropy quartile", fontsize=11)
    ax.set_title(arch_label, fontweight="bold", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.15)

axes[0].set_ylabel("ECE-True (%)", fontsize=11)
axes[1].legend(fontsize=10, loc="upper left")

fig.suptitle(
    "ECE-True by annotation entropy quartile (CIFAR-10H)",
    fontweight="bold", fontsize=13,
)
plt.tight_layout()
out_path = OUT / "fig4_stratified.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved → {out_path}")
