"""
Evaluation metrics for the ambiguous-ground-truth calibration setting.

Both "hard" (single voted label) and "soft" (annotator distribution) variants
are provided for every metric.

Functions
---------
  compute_ece          — Expected Calibration Error (equal-width bins)
  compute_adaptive_ece — Adaptive ECE (equal-mass / quantile bins)
  compute_classwise_ece— Classwise ECE (per-class, then averaged)
  compute_brier        — Brier Score
  compute_nll          — Negative Log-Likelihood
  compute_all_metrics  — Run all metrics, hard + soft, return dict
  stratified_metrics   — Split by ambiguity level and compute ECE-Soft
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# ECE
# ──────────────────────────────────────────────────────────────────────────────

def compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
    min_count: int = 1,
) -> tuple[float, list]:
    """
    Expected Calibration Error (equal-width confidence bins).

    Parameters
    ----------
    probs   : (N, K)  predicted probability vectors
    targets : (N,) int  OR  (N, K) float
              If 1-D integer array  → hard labels (converted to one-hot internally).
              If 2-D float array    → soft label / annotator distribution.
    n_bins  : number of confidence bins
    min_count : bins with fewer examples are skipped

    Returns
    -------
    ece  : scalar ECE value
    info : list of (mean_conf, mean_acc, count) per non-empty bin
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)

    K = probs.shape[1]
    if targets.ndim == 1:
        # Convert integer labels to one-hot
        soft = np.eye(K)[targets.astype(int)]
    else:
        soft = targets.astype(np.float64)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    sacc = soft[np.arange(len(pred)), pred]   # soft accuracy for predicted class

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    info = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        n    = mask.sum()
        if n >= min_count:
            mc = float(conf[mask].mean())
            ma = float(sacc[mask].mean())
            ece += (n / len(probs)) * abs(mc - ma)
            info.append((mc, ma, int(n)))

    return float(ece), info


def compute_adaptive_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
    min_count: int = 1,
) -> float:
    """
    Adaptive ECE with equal-mass (quantile) bins [Nixon et al. 2019].

    Unlike standard ECE which uses equal-width confidence bins, aECE places
    bin boundaries so each bin contains an equal number of samples.  This
    avoids empty bins at extreme confidence values and gives a more reliable
    estimate when the confidence distribution is non-uniform.

    Parameters
    ----------
    probs   : (N, K) predicted probability vectors
    targets : (N,) int  OR  (N, K) float
    n_bins  : number of equal-mass bins
    min_count : bins with fewer examples are skipped
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)

    K = probs.shape[1]
    if targets.ndim == 1:
        soft = np.eye(K)[targets.astype(int)]
    else:
        soft = targets.astype(np.float64)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    sacc = soft[np.arange(len(pred)), pred]

    # Sort by confidence, then split into n_bins equal-mass buckets
    order = np.argsort(conf)
    conf_s = conf[order]
    sacc_s = sacc[order]

    N = len(conf)
    ece = 0.0
    for b in np.array_split(np.arange(N), n_bins):
        if len(b) >= min_count:
            mc = float(conf_s[b].mean())
            ma = float(sacc_s[b].mean())
            ece += (len(b) / N) * abs(mc - ma)
    return float(ece)


def compute_classwise_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
    min_count: int = 1,
) -> float:
    """
    Classwise ECE (cwECE) [Kull et al. 2019].

    For each class k, treat p_k as a "confidence" and the k-th component of
    the target (soft or one-hot) as the "accuracy".  Compute a standard
    equal-width binned ECE for that class, then average over all K classes.

    Parameters
    ----------
    probs   : (N, K) predicted probability vectors
    targets : (N,) int  OR  (N, K) float
    n_bins  : number of confidence bins per class
    min_count : bins with fewer examples are skipped
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)

    N, K = probs.shape
    if targets.ndim == 1:
        soft = np.eye(K)[targets.astype(int)]
    else:
        soft = targets.astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_per_class = []
    for k in range(K):
        p_k = probs[:, k]
        t_k = soft[:, k]
        ece_k = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p_k >= lo) & (p_k < hi)
            n = mask.sum()
            if n >= min_count:
                mc = float(p_k[mask].mean())
                mt = float(t_k[mask].mean())
                ece_k += (n / N) * abs(mc - mt)
        ece_per_class.append(ece_k)
    return float(np.mean(ece_per_class))


def compute_ece_from_conf(
    cal_conf: np.ndarray,
    pred: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE when a non-parametric calibrator provides (cal_conf, pred) instead of probs.

    Parameters
    ----------
    cal_conf : (N,)   calibrated confidence for the predicted class
    pred     : (N,)   predicted class index
    targets  : (N,) int or (N, K) float
    """
    targets = np.asarray(targets)
    N = len(cal_conf)
    K = targets.shape[1] if targets.ndim == 2 else int(targets.max()) + 1
    soft = np.eye(K)[targets.astype(int)] if targets.ndim == 1 else targets.astype(float)
    sacc = soft[np.arange(N), pred.astype(int)]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (cal_conf >= lo) & (cal_conf < hi)
        n    = mask.sum()
        if n > 0:
            ece += (n / N) * abs(float(cal_conf[mask].mean()) - float(sacc[mask].mean()))
    return float(ece)


# ──────────────────────────────────────────────────────────────────────────────
# Brier Score
# ──────────────────────────────────────────────────────────────────────────────

def compute_brier(
    probs: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Multi-class Brier Score: (1/N) Σ_i ‖p̂(xᵢ) − t(xᵢ)‖²

    targets may be int labels or soft distributions.
    """
    probs   = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)
    K = probs.shape[1]
    if targets.ndim == 1:
        soft = np.eye(K)[targets.astype(int)]
    else:
        soft = targets.astype(np.float64)
    return float(np.mean(np.sum((probs - soft) ** 2, axis=1)))


# ──────────────────────────────────────────────────────────────────────────────
# NLL
# ──────────────────────────────────────────────────────────────────────────────

def compute_nll(
    probs: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Negative Log-Likelihood: −(1/N) Σ_i Σ_k t_k(xᵢ) log p̂_k(xᵢ)

    For hard targets this reduces to standard cross-entropy with hard labels.
    """
    probs   = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets)
    K = probs.shape[1]
    if targets.ndim == 1:
        soft = np.eye(K)[targets.astype(int)]
    else:
        soft = targets.astype(np.float64)
    log_p = np.log(np.clip(probs, eps, 1.0))
    return float(-np.mean((soft * log_p).sum(axis=1)))


# ──────────────────────────────────────────────────────────────────────────────
# Composite helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_ece_sampled(
    probs: np.ndarray,
    labels_soft: np.ndarray,
    n_bins: int = 15,
    n_samples: int = 100,
    seed: int = 0,
) -> float:
    """
    ECE against sampled hard labels: for each sample, draw a hard label from
    its soft label distribution π, then compute ECE.  Repeat n_samples times
    and return the average.

    This metric captures what happens when each sample has a *single* annotator
    whose label is drawn from the annotator distribution π(x).
    """
    rng = np.random.default_rng(seed)
    N, K = labels_soft.shape
    ece_sum = 0.0
    for _ in range(n_samples):
        sampled = np.array([rng.choice(K, p=labels_soft[i]) for i in range(N)])
        ece_i, _ = compute_ece(probs, sampled, n_bins=n_bins)
        ece_sum += ece_i
    return ece_sum / n_samples


def compute_brier_sampled(
    probs: np.ndarray,
    labels_soft: np.ndarray,
) -> float:
    """
    Brier Score against "true" (sampled) labels:
        E_{y~π} [||p - e_y||²] = Σ_k π_k · ||p - e_k||²

    Closed-form: = ||p||² + 1 - 2·p·π  (per sample, then averaged).
    This differs from brier_soft = ||p - π||².
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels_soft = np.asarray(labels_soft, dtype=np.float64)
    # Per sample: Σ_k π_k (Σ_j (p_j - δ_{jk})^2)
    #           = Σ_k π_k (||p||^2 - 2p_k + 1)
    #           = ||p||^2 + 1 - 2 Σ_k π_k p_k
    p_sq = np.sum(probs ** 2, axis=1)          # ||p||^2
    dot  = np.sum(probs * labels_soft, axis=1)  # p · π
    return float(np.mean(p_sq + 1.0 - 2.0 * dot))


def compute_nll_sampled(
    probs: np.ndarray,
    labels_soft: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    NLL against "true" (sampled) labels:
        E_{y~π} [-log p_y] = -Σ_k π_k log p_k

    This is mathematically identical to nll_soft (cross-entropy with soft targets).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels_soft = np.asarray(labels_soft, dtype=np.float64)
    log_p = np.log(np.clip(probs, eps, 1.0))
    return float(-np.mean((labels_soft * log_p).sum(axis=1)))


def compute_all_metrics(
    probs: np.ndarray,
    labels_hard: np.ndarray,
    labels_soft: np.ndarray,
    n_bins: int = 15,
    name: str = "",
) -> dict:
    """
    Compute all metrics (hard + soft + sampled) for a single calibration output.

    Returns a flat dict with keys:
      ece_hard, ece_soft, ece_sampled,
      adaptive_ece_true, classwise_ece_true,
      brier_hard, brier_soft, brier_sampled,
      nll_hard, nll_soft, nll_sampled
    """
    ece_h, _ = compute_ece(probs, labels_hard, n_bins=n_bins)
    ece_s, _ = compute_ece(probs, labels_soft, n_bins=n_bins)
    ece_samp = compute_ece_sampled(probs, labels_soft, n_bins=n_bins)
    return {
        "name":               name,
        "ece_hard":           ece_h,
        "ece_soft":           ece_s,
        "ece_sampled":        ece_samp,
        "adaptive_ece_true":  compute_adaptive_ece(probs, labels_soft, n_bins=n_bins),
        "classwise_ece_true": compute_classwise_ece(probs, labels_soft, n_bins=n_bins),
        "brier_hard":         compute_brier(probs, labels_hard),
        "brier_soft":         compute_brier(probs, labels_soft),
        "brier_sampled":      compute_brier_sampled(probs, labels_soft),
        "nll_hard":           compute_nll(probs, labels_hard),
        "nll_soft":           compute_nll(probs, labels_soft),
        "nll_sampled":        compute_nll_sampled(probs, labels_soft),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Stratified metrics
# ──────────────────────────────────────────────────────────────────────────────

def annotation_entropy(labels_soft: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy H(π(x)) for each sample — measures ambiguity level."""
    lsoft = np.clip(labels_soft, eps, 1.0)
    return -np.sum(lsoft * np.log(lsoft), axis=1)


def stratified_ece_soft(
    probs: np.ndarray,
    labels_soft: np.ndarray,
    n_bins: int = 15,
    n_quantiles: int = 4,
) -> list[dict]:
    """
    Compute ECE-Soft stratified by ambiguity level (entropy of π(x)).

    Returns a list of dicts with keys: quantile, entropy_lo, entropy_hi, ece_soft, n
    """
    H = annotation_entropy(labels_soft)
    quantile_edges = np.quantile(H, np.linspace(0, 1, n_quantiles + 1))

    results = []
    for q in range(n_quantiles):
        lo, hi = quantile_edges[q], quantile_edges[q + 1]
        mask = (H >= lo) & (H <= hi) if q == n_quantiles - 1 else (H >= lo) & (H < hi)
        if mask.sum() > 0:
            ece_s, _ = compute_ece(probs[mask], labels_soft[mask], n_bins=n_bins)
            results.append({
                "quantile":    q + 1,
                "entropy_lo":  float(lo),
                "entropy_hi":  float(hi),
                "ece_soft":    ece_s,
                "n":           int(mask.sum()),
            })
    return results


def ambiguity_split_ece(
    probs: np.ndarray,
    labels_hard: np.ndarray,
    labels_soft: np.ndarray,
    entropy_thresh: Optional[float] = None,
    n_bins: int = 15,
) -> dict:
    """
    Split into ambiguous / unambiguous groups and compute ECE-Soft for each.

    entropy_thresh : if None, uses the median entropy as threshold.
    """
    H = annotation_entropy(labels_soft)
    if entropy_thresh is None:
        entropy_thresh = float(np.median(H))

    amb  = H >= entropy_thresh
    unab = ~amb

    res = {"entropy_threshold": entropy_thresh}
    for tag, mask in [("ambiguous", amb), ("clear", unab)]:
        if mask.sum() > 0:
            ece_s, _ = compute_ece(probs[mask], labels_soft[mask], n_bins=n_bins)
            res[f"ece_soft_{tag}"]  = ece_s
            res[f"n_{tag}"]         = int(mask.sum())
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Pretty print
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list[dict]) -> None:
    # --- main ECE / Brier / NLL table ---
    header = (f"{'Method':<22} | {'ECE-Hard':>9} | {'ECE-True':>9} | {'ECE-Soft':>9} | "
              f"{'Br-Hard':>8} | {'Br-True':>8} | {'Br-Soft':>8} | "
              f"{'NLL-Hard':>9} | {'NLL-True':>9} | {'NLL-Soft':>9}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        ece_samp = r.get('ece_sampled', float('nan'))
        br_samp  = r.get('brier_sampled', float('nan'))
        nll_samp = r.get('nll_sampled', float('nan'))
        print(
            f"{r['name']:<22} | "
            f"{r['ece_hard']*100:>8.2f}% | "
            f"{ece_samp*100:>8.2f}% | "
            f"{r['ece_soft']*100:>8.2f}% | "
            f"{r['brier_hard']:>8.4f} | "
            f"{br_samp:>8.4f} | "
            f"{r['brier_soft']:>8.4f} | "
            f"{r['nll_hard']:>9.4f} | "
            f"{nll_samp:>9.4f} | "
            f"{r['nll_soft']:>9.4f}"
        )
    print("=" * len(header))

    # --- Adaptive ECE + Classwise ECE (true-label variants) ---
    header2 = (f"{'Method':<22} | {'aECE-True':>10} | {'cwECE-True':>11}")
    print()
    print("=" * len(header2))
    print(header2)
    print("-" * len(header2))
    for r in results:
        aece  = r.get('adaptive_ece_true',  float('nan'))
        cwece = r.get('classwise_ece_true', float('nan'))
        print(
            f"{r['name']:<22} | "
            f"{aece*100:>9.2f}% | "
            f"{cwece*100:>10.2f}%"
        )
    print("=" * len(header2))
