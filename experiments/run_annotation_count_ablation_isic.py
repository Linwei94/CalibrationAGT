"""
Annotation-count ablation for ambiguity-aware calibrators on ISIC 2019.

Subsamples m annotations per calibration example (m = 1..9) from the
synthetic dermatologist annotation model to measure how annotation richness
affects post-hoc calibration.

Methods covered
---------------
  SLTS, MCTS, VS, SoftPlatt, Dirichlet-Soft, HB-Soft, IR-Soft

Usage
-----
    python run_annotation_count_ablation_isic.py --arch efficientnet_b4
    python run_annotation_count_ablation_isic.py --arch vit_s16
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
    MonteCarloTS,
    SoftHistogramBinning,
    SoftIsotonicRegression,
    SoftLabelTS,
    SoftPlattScaling,
    VectorScaling,
)
from metrics import compute_all_metrics
from run_isic2019 import generate_soft_labels


METHOD_ORDER = [
    "SLTS",
    "MCTS",
    "VS",
    "SoftPlatt",
    "Dirichlet-Soft",
    "HB-Soft",
    "IR-Soft",
]

PLOT_COLORS = {
    "SLTS": "#4CAF80",
    "MCTS": "#9B59B6",
    "VS": "#F39C12",
    "SoftPlatt": "#4C78A8",
    "Dirichlet-Soft": "#D45B3A",
    "HB-Soft": "#2A9D8F",
    "IR-Soft": "#34495E",
}

N_ANNOTATORS_FULL = 9


def sample_empirical_distributions(labels_soft: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Sample m annotations per example and return the empirical distribution."""
    counts = np.empty_like(labels_soft)
    for i, p in enumerate(labels_soft):
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
        p /= p.sum()
        counts[i] = rng.multinomial(m, p)
    return counts / float(m)


def fit_and_eval(
    method: str,
    logits_cal: np.ndarray,
    probs_cal: np.ndarray,
    logits_te: np.ndarray,
    labels_hard_te: np.ndarray,
    labels_soft_cal_m: np.ndarray,
    labels_soft_te_true: np.ndarray,
    n_bins: int,
    seed: int,
    mcts_samples: int,
    adam_epochs: int,
    device: torch.device,
) -> dict:
    logits_cal_t = torch.tensor(logits_cal, dtype=torch.float32, device=device)
    labels_soft_cal_t = torch.tensor(labels_soft_cal_m, dtype=torch.float32, device=device)

    def apply_parametric_device(calibrator, logits_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(logits_np, dtype=torch.float32, device=device)
            p = torch.softmax(calibrator(t), dim=1)
        return p.detach().cpu().numpy()

    if method == "SLTS":
        cal = SoftLabelTS().to(device).fit(logits_cal_t, labels_soft_cal_t)
        probs = apply_parametric_device(cal, logits_te)
        temperature = cal.T
    elif method == "MCTS":
        cal = MonteCarloTS(n_samples=mcts_samples, seed=seed).to(device).fit(logits_cal_t, labels_soft_cal_t)
        probs = apply_parametric_device(cal, logits_te)
        temperature = cal.T
    elif method == "VS":
        cal = VectorScaling(logits_cal.shape[1]).to(device).fit(logits_cal_t, labels_soft_cal_t, n_epochs=adam_epochs)
        probs = apply_parametric_device(cal, logits_te)
        temperature = None
    elif method == "SoftPlatt":
        cal = SoftPlattScaling(logits_cal.shape[1], n_epochs=adam_epochs).to(device).fit(logits_cal_t, labels_soft_cal_t)
        probs = apply_parametric_device(cal, logits_te)
        temperature = None
    elif method == "Dirichlet-Soft":
        cal = DirichletCalibration(logits_cal.shape[1], n_epochs=adam_epochs).to(device).fit_soft(logits_cal_t, labels_soft_cal_t)
        probs = apply_parametric_device(cal, logits_te)
        temperature = None
    elif method == "HB-Soft":
        hb = SoftHistogramBinning(n_bins=n_bins).fit(probs_cal, labels_soft_cal_m)
        conf, pred = hb.calibrate(torch.softmax(torch.tensor(logits_te), dim=1).numpy())
        probs = np.zeros((len(pred), logits_cal.shape[1]), dtype=np.float64)
        for i in range(len(pred)):
            probs[i, pred[i]] = conf[i]
            rest = (1.0 - conf[i]) / (probs.shape[1] - 1)
            probs[i] += rest
            probs[i, pred[i]] = conf[i]
        temperature = None
    elif method == "IR-Soft":
        ir = SoftIsotonicRegression().fit(probs_cal, labels_soft_cal_m)
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

    metrics = compute_all_metrics(
        probs,
        labels_hard_te,
        labels_soft_te_true,
        n_bins=n_bins,
        name=method,
    )
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

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    m_values = results["m_values"]
    ts_ref = results.get("ts_reference_ece_true")
    if ts_ref is not None:
        ax.axhline(ts_ref * 100.0, color="#777777", lw=1.5, ls="--", alpha=0.8, label="TS")
    for method in METHOD_ORDER:
        if method not in results["methods"]:
            continue
        trace = results["methods"][method]
        y = [trace[str(m)]["ece_true"] * 100 for m in m_values]
        ax.plot(m_values, y, label=method, color=PLOT_COLORS[method], lw=2.0, marker="o", markersize=5)

    ax.set_xlabel("Annotations per calibration example (m)")
    ax.set_ylabel("ECE_true (%)")
    ax.set_title(f"Annotation-count ablation on ISIC 2019 ({results['arch']})")
    ax.set_xticks(m_values)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["efficientnet_b4", "vit_s16"], default="efficientnet_b4")
    p.add_argument("--cache-dir", default="experiments/cache")
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--figures-dir", default="experiments/figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--m-min", type=int, default=1)
    p.add_argument("--m-max", type=int, default=9)
    p.add_argument("--mcts-samples", type=int, default=50)
    p.add_argument("--adam-epochs", type=int, default=400)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── Load cached logits and labels ─────────────────────────────────────────
    logits_cal = np.load(cache_dir / f"isic2019_logits_val_{args.arch}.npy")
    logits_te = np.load(cache_dir / f"isic2019_logits_test_{args.arch}.npy")
    labels_hard_cal = np.load(cache_dir / "isic2019_labels_val.npy")
    labels_hard_te = np.load(cache_dir / "isic2019_labels_test.npy")

    probs_cal = torch.softmax(torch.tensor(logits_cal), dim=1).numpy()

    # Full soft labels (9 annotators) used as "true" soft targets
    labels_soft_cal_true = generate_soft_labels(labels_hard_cal, n_annotators=N_ANNOTATORS_FULL, seed=args.seed)
    labels_soft_te_true = generate_soft_labels(labels_hard_te, n_annotators=N_ANNOTATORS_FULL, seed=args.seed)

    print(f"Cal n={len(logits_cal)}, Test n={len(logits_te)}")

    # ── TS reference ──────────────────────────────────────────────────────────
    ts_reference = None
    ref_path = results_dir / f"isic2019_results_{args.arch}.json"
    if ref_path.exists():
        with open(ref_path) as f:
            ref = json.load(f)
        for row in ref.get("main_results", []):
            if row.get("name") == "TS":
                ts_reference = row["ece_sampled"]
                break

    # ── Main ablation loop ────────────────────────────────────────────────────
    m_values = list(range(args.m_min, args.m_max + 1))
    results = {
        "dataset": "ISIC2019",
        "arch": args.arch,
        "seed": args.seed,
        "ts_reference_ece_true": ts_reference,
        "m_values": m_values,
        "methods": {method: {} for method in METHOD_ORDER},
    }

    for m in m_values:
        print(f"\n=== m = {m} ===")
        labels_soft_cal_m = sample_empirical_distributions(labels_soft_cal_true, m, rng)
        for method in METHOD_ORDER:
            metrics = fit_and_eval(
                method=method,
                logits_cal=logits_cal,
                probs_cal=probs_cal,
                logits_te=logits_te,
                labels_hard_te=labels_hard_te,
                labels_soft_cal_m=labels_soft_cal_m,
                labels_soft_te_true=labels_soft_te_true,
                n_bins=args.n_bins,
                seed=args.seed + m,
                mcts_samples=args.mcts_samples,
                adam_epochs=args.adam_epochs,
                device=device,
            )
            results["methods"][method][str(m)] = metrics
            print(f"  {method:<15} ECE_true={metrics['ece_true']*100:5.2f}%")

    suffix = f"_{args.tag}" if args.tag else ""
    out_json = results_dir / f"annotation_count_ablation_isic_{args.arch}{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {out_json}")

    out_pdf = figures_dir / f"annotation_count_ablation_isic_{args.arch}{suffix}.pdf"
    out_png = figures_dir / f"annotation_count_ablation_isic_{args.arch}{suffix}.png"
    make_plot(results, out_pdf, out_png)
    print(f"Saved plot → {out_pdf}")


if __name__ == "__main__":
    main()
