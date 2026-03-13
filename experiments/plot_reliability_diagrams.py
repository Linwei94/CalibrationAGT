"""
Reliability diagrams for ALL calibration methods.

Produces two figures per dataset/arch:
  (1) 1×5 summary panel: Uncal / TS / SLTS / Dirichlet-Soft / IR-Soft
  (2) 3×5 full panel:    all 13 methods (hard-label, annotation-free, soft)

Datasets: cifar10h, chaosnli, isic2019, dermamnist

Usage
-----
    python plot_reliability_diagrams.py --dataset cifar10h --arch resnet50
    python plot_reliability_diagrams.py --dataset chaosnli --arch roberta_large
    python plot_reliability_diagrams.py --dataset isic2019 --arch efficientnet_b4
    python plot_reliability_diagrams.py --dataset dermamnist --arch resnet18
"""

from __future__ import annotations

import argparse
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
    HardHistogramBinning,
    LabelSmoothTS,
    MonteCarloTS,
    PlattScaling,
    SoftHistogramBinning,
    SoftIsotonicRegression,
    SoftLabelTS,
    SoftPlattScaling,
    TemperatureScaling,
    VectorScaling,
)
from run_chaosnli import download_chaosnli, parse_chaosnli
from run_cifar10h import download_cifar10h, get_cifar10_testset


CONFUSION_ISIC = np.array([
    [0.73, 0.14, 0.02, 0.03, 0.08, 0.00, 0.00, 0.00],
    [0.15, 0.76, 0.01, 0.01, 0.06, 0.01, 0.00, 0.00],
    [0.02, 0.01, 0.81, 0.05, 0.07, 0.01, 0.01, 0.02],
    [0.03, 0.01, 0.04, 0.65, 0.11, 0.00, 0.00, 0.16],
    [0.12, 0.05, 0.03, 0.10, 0.62, 0.00, 0.00, 0.08],
    [0.01, 0.02, 0.02, 0.01, 0.02, 0.87, 0.03, 0.02],
    [0.00, 0.01, 0.02, 0.01, 0.01, 0.02, 0.91, 0.02],
    [0.01, 0.01, 0.03, 0.18, 0.09, 0.00, 0.01, 0.67],
], dtype=np.float64)

CONFUSION_DERM = np.array([
    [0.62, 0.07, 0.16, 0.02, 0.07, 0.05, 0.01],  # AK
    [0.07, 0.73, 0.09, 0.03, 0.04, 0.03, 0.01],  # BCC
    [0.12, 0.06, 0.62, 0.05, 0.08, 0.06, 0.01],  # BKL
    [0.02, 0.03, 0.04, 0.83, 0.03, 0.04, 0.01],  # DF
    [0.04, 0.04, 0.07, 0.02, 0.63, 0.19, 0.01],  # Mel
    [0.03, 0.02, 0.06, 0.04, 0.14, 0.70, 0.01],  # NV
    [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.92],  # Vasc
], dtype=np.float64)


def _generate_soft_labels(hard_labels, n_annotators, confusion_matrix, seed=42):
    rng = np.random.default_rng(seed)
    N = len(hard_labels)
    K = confusion_matrix.shape[0]
    soft = np.zeros((N, K), dtype=np.float32)
    for i, y in enumerate(hard_labels):
        probs = confusion_matrix[y]
        annotations = rng.choice(K, size=n_annotators, p=probs)
        for a in annotations:
            soft[i, a] += 1.0
        soft[i] /= n_annotators
    return soft


# ── Colour palette (copied from figures/generate_all.py) ─────────────────────
PAL = dict(
    uncal="#5A8FC2", ts="#E07040", ps="#B06090",
    mcts="#9B59B6", slts="#4CAF80", vs="#F39C12",
    hbh="#888888", ir="#34495E",
    gap="#C0392B", gap_over="#FFB3B3", gap_under="#C8F0C8",
    dirichlet="#D45B3A", lsts="#66BB6A", hbs="#26A69A",
)

# All 13 methods: (key, display_name, color)
ALL_METHODS = [
    # Voted-label baselines
    ("Uncal",          "Uncal",         PAL["uncal"]),
    ("TS",             "TS",            PAL["ts"]),
    ("Platt",          "Platt (PS)",    PAL["ps"]),
    ("Dirichlet-Hard", "Dir.-Hard",     "#A0522D"),
    ("HB-Hard",        "HB-Hard",       PAL["hbh"]),
    # Annotation-free
    ("LS-TS",          "LS-TS",         PAL["lsts"]),
    # Soft methods
    ("SLTS",           "SLTS",          PAL["slts"]),
    ("MCTS",           "MCTS",          PAL["mcts"]),
    ("SoftPlatt",      "SoftPlatt",     "#7B68EE"),
    ("VS",             "VS",            PAL["vs"]),
    ("Dirichlet-Soft", "Dir.-Soft",     PAL["dirichlet"]),
    ("HB-Soft",        "HB-Soft",       PAL["hbs"]),
    ("IR-Soft",        "IR-Soft",       PAL["ir"]),
]

# Summary 4-method set for main figure
SUMMARY_METHODS = ["Uncal", "TS", "SLTS", "Dirichlet-Soft"]


# ── ECE + bin info ────────────────────────────────────────────────────────────
def ece_bins(probs, targets, n_bins=12, min_count=3):
    K = probs.shape[1]
    soft = np.eye(K)[targets.astype(int)] if targets.ndim == 1 else targets.astype(float)
    conf = probs.max(1)
    pred = probs.argmax(1)
    sacc = soft[np.arange(len(pred)), pred]
    edges = np.linspace(0, 1, n_bins + 1)
    ece, info = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        n = m.sum()
        if n >= min_count:
            mc = float(conf[m].mean())
            ma = float(sacc[m].mean())
            ece += (n / len(probs)) * abs(mc - ma)
            info.append((mc, ma, int(n)))
    return float(ece), info


# ── Reliability diagram axis ──────────────────────────────────────────────────
def rel_ax(ax, probs, targets, title, color, n_bins=12, arrow=False):
    ece, info = ece_bins(probs, targets, n_bins=n_bins)
    if not info:
        ax.set_title(f"{title}\nECE={ece*100:.1f}%")
        return ece
    confs = np.array([b[0] for b in info])
    accs = np.array([b[1] for b in info])
    w = 0.9 / n_bins
    for c, a in zip(confs, accs):
        col = PAL["gap_under"] if a > c else PAL["gap_over"]
        ax.fill_between([c - w / 2, c + w / 2], [min(c, a)] * 2, [max(c, a)] * 2,
                        color=col, alpha=0.5, zorder=1)
    ax.bar(confs, accs, width=w * 0.80, color=color, alpha=0.85, zorder=2)
    ax.plot([0, 1], [0, 1], "--", color="#444", lw=1.3, zorder=3)
    if arrow and info:
        idx = int(np.argmax(np.abs(confs - accs)))
        c0, a0 = confs[idx], accs[idx]
        ax.annotate("", xy=(c0, a0), xytext=(c0, c0),
                    arrowprops=dict(arrowstyle="<->", color=PAL["gap"], lw=2.0))
        mid_y = (c0 + a0) / 2
        if c0 > 0.65:
            ax.text(c0 - .22, mid_y, f"Gap {c0 - a0:+.2f}",
                    color=PAL["gap"], fontsize=7.5, va="center", ha="right",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
        else:
            ax.text(c0 + .04, mid_y, f"Gap {c0 - a0:+.2f}",
                    color=PAL["gap"], fontsize=8, va="center", ha="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_title(f"{title}\nECE = {ece*100:.1f}%", fontweight="bold", fontsize=9)
    ax.grid(True, alpha=0.2)
    return ece


# ── Data loading ──────────────────────────────────────────────────────────────
def load_split(cache_dir, dataset, arch, seed):
    if dataset == "cifar10h":
        soft_all = download_cifar10h(str(cache_dir))
        logits_all = np.load(cache_dir / f"logits_test_{arch}.npy")
        testset = get_cifar10_testset(str(cache_dir / "cifar10"))
        hard_all = np.array([y for _, y in testset])
        rng = np.random.default_rng(seed)
        idx = np.arange(len(soft_all))
        K = soft_all.shape[1]
        cal_parts, te_parts = [], []
        for k in range(K):
            arr = idx[hard_all == k]
            perm = rng.permutation(arr)
            cut = len(arr) // 2
            cal_parts.append(perm[:cut])
            te_parts.append(perm[cut:])
        idx_cal = np.sort(np.concatenate(cal_parts))
        idx_te = np.sort(np.concatenate(te_parts))
        return (
            logits_all[idx_cal], logits_all[idx_te],
            hard_all[idx_cal], hard_all[idx_te],
            soft_all[idx_cal], soft_all[idx_te],
        )
    elif dataset == "chaosnli":
        raw = download_chaosnli(str(cache_dir), "snli") + download_chaosnli(str(cache_dir), "mnli")
        _, soft_all, hard_all = parse_chaosnli(raw)
        logits_all = np.load(cache_dir / f"logits_chaosnli_combined_{arch}.npy")
        N = len(hard_all)
        rng = np.random.default_rng(seed)
        idx = np.arange(N)
        mask = np.zeros(N, dtype=bool)
        for c in range(3):
            c_idx = idx[hard_all == c]
            mask[rng.choice(c_idx, size=len(c_idx) // 2, replace=False)] = True
        idx_cal, idx_te = idx[mask], idx[~mask]
        return (
            logits_all[idx_cal], logits_all[idx_te],
            hard_all[idx_cal], hard_all[idx_te],
            soft_all[idx_cal], soft_all[idx_te],
        )
    elif dataset == "isic2019":
        logits_cal = np.load(cache_dir / f"isic2019_logits_val_{arch}.npy")
        logits_te = np.load(cache_dir / f"isic2019_logits_test_{arch}.npy")
        hard_cal = np.load(cache_dir / "isic2019_labels_val.npy")
        hard_te = np.load(cache_dir / "isic2019_labels_test.npy")
        soft_cal = _generate_soft_labels(hard_cal, 9, CONFUSION_ISIC, seed=seed)
        soft_te = _generate_soft_labels(hard_te, 9, CONFUSION_ISIC, seed=seed)
        return (logits_cal, logits_te, hard_cal, hard_te, soft_cal, soft_te)
    elif dataset == "dermamnist":
        import sys as _sys
        _sys.path.insert(0, str(cache_dir.parent))
        from medmnist import DermaMNIST as _DermaMNISTDS
        ds_val = _DermaMNISTDS(split="val", download=True, root=str(cache_dir))
        ds_te = _DermaMNISTDS(split="test", download=True, root=str(cache_dir))
        hard_cal = ds_val.labels.flatten().astype(int)
        hard_te = ds_te.labels.flatten().astype(int)
        logits_cal = np.load(cache_dir / f"dermamnist_logits_val_{arch}.npy")
        logits_te = np.load(cache_dir / f"dermamnist_logits_test_{arch}.npy")
        soft_cal = _generate_soft_labels(hard_cal, 5, CONFUSION_DERM, seed=seed)
        soft_te = _generate_soft_labels(hard_te, 5, CONFUSION_DERM, seed=seed)
        return (logits_cal, logits_te, hard_cal, hard_te, soft_cal, soft_te)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── Fit all calibrators ───────────────────────────────────────────────────────
def fit_all(logits_cal, logits_te, hard_cal, soft_cal, adam_epochs, mcts_samples, device):
    lc = torch.tensor(logits_cal, dtype=torch.float32, device=device)
    sc = torch.tensor(soft_cal, dtype=torch.float32, device=device)
    hc = torch.tensor(hard_cal, dtype=torch.long, device=device)
    K = logits_cal.shape[1]

    def sm(logits_np):
        with torch.no_grad():
            return torch.softmax(torch.tensor(logits_np, dtype=torch.float32, device=device), dim=1).cpu().numpy()

    def cal(m, logits_np):
        with torch.no_grad():
            return torch.softmax(m(torch.tensor(logits_np, dtype=torch.float32, device=device)), dim=1).cpu().numpy()

    def nonparam_probs(conf, pred, K):
        p = np.zeros((len(pred), K), dtype=np.float64)
        for i in range(len(pred)):
            p[i, pred[i]] = conf[i]
            rest = (1.0 - conf[i]) / (K - 1)
            p[i] += rest
            p[i, pred[i]] = conf[i]
        return p

    probs_cal = sm(logits_cal)
    result = {}
    result["Uncal"] = sm(logits_te)

    # Voted-label baselines
    ts = TemperatureScaling().to(device).fit(lc, hc)
    result["TS"] = cal(ts, logits_te)

    ps = PlattScaling(K).to(device).fit(lc, hc)
    result["Platt"] = cal(ps, logits_te)

    dh = DirichletCalibration(K, n_epochs=adam_epochs).to(device).fit_hard(lc, hc)
    result["Dirichlet-Hard"] = cal(dh, logits_te)

    hbh = HardHistogramBinning(n_bins=15).fit(probs_cal, hard_cal)
    conf_h, pred_h = hbh.calibrate(sm(logits_te))
    result["HB-Hard"] = nonparam_probs(conf_h, pred_h, K)

    # Annotation-free
    lsts = LabelSmoothTS().to(device).fit(lc, hc)
    result["LS-TS"] = cal(lsts, logits_te)

    # Soft methods
    slts = SoftLabelTS().to(device).fit(lc, sc)
    result["SLTS"] = cal(slts, logits_te)

    mcts = MonteCarloTS(n_samples=mcts_samples, seed=42).to(device).fit(lc, sc)
    result["MCTS"] = cal(mcts, logits_te)

    spl = SoftPlattScaling(K, n_epochs=adam_epochs).to(device).fit(lc, sc)
    result["SoftPlatt"] = cal(spl, logits_te)

    vs = VectorScaling(K).to(device).fit(lc, sc, n_epochs=adam_epochs)
    result["VS"] = cal(vs, logits_te)

    ds = DirichletCalibration(K, n_epochs=adam_epochs).to(device).fit_soft(lc, sc)
    result["Dirichlet-Soft"] = cal(ds, logits_te)

    hbs = SoftHistogramBinning(n_bins=15).fit(probs_cal, soft_cal)
    conf_s, pred_s = hbs.calibrate(sm(logits_te))
    result["HB-Soft"] = nonparam_probs(conf_s, pred_s, K)

    ir = SoftIsotonicRegression().fit(probs_cal, soft_cal)
    conf_i, pred_i = ir.calibrate(sm(logits_te))
    result["IR-Soft"] = nonparam_probs(conf_i, pred_i, K)

    return result


def plot_summary(all_probs, soft_te, dataset, arch, n_bins, figures_dir, suffix):
    """1×5 summary panel (5 key methods)."""
    n = len(SUMMARY_METHODS)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 3.5))
    for ax, key in zip(axes, SUMMARY_METHODS):
        _, name, color = next(m for m in ALL_METHODS if m[0] == key)
        rel_ax(ax, all_probs[key], soft_te, title=name, color=color,
               n_bins=n_bins, arrow=(key == "Uncal"))
    fig.tight_layout()
    out = figures_dir / f"reliability_diagrams_{dataset}_{arch}{suffix}"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary → {out}.pdf")


def plot_full(all_probs, soft_te, dataset, arch, n_bins, figures_dir, suffix):
    """3×5 full panel (all 13 methods + 2 blank)."""
    ncols, nrows = 5, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12))
    axes_flat = axes.flatten()

    for i, (key, name, color) in enumerate(ALL_METHODS):
        ax = axes_flat[i]
        rel_ax(ax, all_probs[key], soft_te, title=name, color=color,
               n_bins=n_bins, arrow=(key == "Uncal"))

    # Hide unused axes
    for j in range(len(ALL_METHODS), nrows * ncols):
        axes_flat[j].set_visible(False)

    # Row labels
    row_labels = ["Voted-label baselines", "Annotation-free / Soft (TS-based)", "Soft (non-parametric / matrix)"]
    row_starts = [0, 5, 10]  # method indices where each group starts
    for row_idx, (label, start) in enumerate(zip(row_labels, row_starts)):
        axes[row_idx, 0].set_ylabel(f"{label}\n\nAccuracy", fontsize=8.5, labelpad=10)

    fig.tight_layout()
    out = figures_dir / f"reliability_diagrams_full_{dataset}_{arch}{suffix}"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved full → {out}.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10h", "chaosnli", "isic2019", "dermamnist"], default="cifar10h")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bins", type=int, default=12)
    p.add_argument("--adam-epochs", type=int, default=400)
    p.add_argument("--mcts-samples", type=int, default=50)
    p.add_argument("--cache-dir", default="experiments/cache")
    p.add_argument("--figures-dir", default="experiments/figures")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(args.cache_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    print(f"Loading {args.dataset} / {args.arch} ...")
    logits_cal, logits_te, hard_cal, hard_te, soft_cal, soft_te = \
        load_split(cache_dir, args.dataset, args.arch, args.seed)

    print("Fitting all calibrators ...")
    all_probs = fit_all(logits_cal, logits_te, hard_cal, soft_cal,
                        args.adam_epochs, args.mcts_samples, device)

    suffix = f"_{args.tag}" if args.tag else ""
    plot_summary(all_probs, soft_te, args.dataset, args.arch, args.n_bins, figures_dir, suffix)
    plot_full(all_probs, soft_te, args.dataset, args.arch, args.n_bins, figures_dir, suffix)


if __name__ == "__main__":
    main()
