"""
Calibration methods for the ambiguous ground truth setting.

Methods
-------
  TemperatureScaling  (TS)   — NLL vs. hard (voted) labels          [Guo et al. 2017]
  PlattScaling        (PS)   — matrix scaling (W,b) vs. hard labels [Platt 1999 / Kull 2019]
  SoftLabelTS         (SLTS) — KL(π(x) ‖ softmax(z/T))             [ours]
  MonteCarloTS        (MCTS) — sample individual annotations → NLL  [ours, cf. Stutz 2023]
  VectorScaling       (VS)   — per-class temperatures, soft labels  [ours]
  HardHistogramBinning(HB-H) — non-parametric, hard accuracy targets[Zadrozny & Elkan 2001]
  SoftHistogramBinning(HB-S) — non-parametric, soft targets         [ours]
  SoftIsotonicReg     (IR)   — isotonic regression, soft targets    [ours]
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression as SkIso


# ──────────────────────────────────────────────────────────────────────────────
# Parametric methods
# ──────────────────────────────────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """Standard TS: minimise NLL against hard (voted/majority) labels."""

    def __init__(self, init_T: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels_hard: torch.Tensor) -> "TemperatureScaling":
        """
        Parameters
        ----------
        logits      : (N, K) pre-softmax logits
        labels_hard : (N,)   integer class labels
        """
        opt  = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                  tolerance_grad=1e-9, tolerance_change=1e-11)
        crit = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = crit(self(logits), labels_hard)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @property
    def T(self) -> float:
        return self.temperature.item()


class PlattScaling(nn.Module):
    """
    Multiclass Platt / Matrix Scaling [Platt 1999; Kull et al. 2019].

    Learns an affine transformation of the logit vector before softmax:
        calibrated_logits = W · z + b
    where W is a diagonal matrix (per-class weight) and b is a per-class bias.
    This strictly generalises Temperature Scaling (TS uses W = (1/T)·I, b = 0)
    and is calibrated against *hard* (voted) labels.

    Using a diagonal W (rather than full matrix) avoids over-fitting on small
    calibration sets while still enabling per-class adjustment.
    """

    def __init__(self, n_classes: int, lr: float = 0.01, n_epochs: int = 2000):
        super().__init__()
        self.W = nn.Parameter(torch.ones(n_classes))   # diagonal of weight matrix
        self.b = nn.Parameter(torch.zeros(n_classes))  # bias per class
        self._lr      = lr
        self._n_epochs = n_epochs

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.W + self.b

    def fit(self, logits: torch.Tensor, labels_hard: torch.Tensor) -> "PlattScaling":
        """
        Parameters
        ----------
        logits      : (N, K)
        labels_hard : (N,) integer class labels
        """
        opt  = torch.optim.Adam([self.W, self.b], lr=self._lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()

        for _ in range(self._n_epochs):
            opt.zero_grad()
            loss = crit(self(logits), labels_hard)
            loss.backward()
            opt.step()

        return self


class HardHistogramBinning:
    """
    Standard Histogram Binning with *hard* accuracy targets [Zadrozny & Elkan 2001].

    In each confidence bin, the calibration target is the fraction of examples
    where the predicted class equals the voted (hard) label.  This is the
    classic baseline against which our soft-target variant is compared.
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins    = n_bins
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.bin_vals  = np.zeros(n_bins)

    def fit(self, probs: np.ndarray, labels_hard: np.ndarray) -> "HardHistogramBinning":
        """
        Parameters
        ----------
        probs       : (N, K) predicted probabilities
        labels_hard : (N,)   integer voted labels
        """
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == labels_hard).astype(float)

        for i, (lo, hi) in enumerate(zip(self.bin_edges[:-1], self.bin_edges[1:])):
            mask = (conf >= lo) & (conf < hi)
            self.bin_vals[i] = correct[mask].mean() if mask.sum() > 0 else (lo + hi) / 2.0
        return self

    def calibrate(self, probs: np.ndarray):
        """Returns (calibrated_conf, predicted_class) arrays."""
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        cal  = np.zeros_like(conf)
        for i, (lo, hi) in enumerate(zip(self.bin_edges[:-1], self.bin_edges[1:])):
            mask = (conf >= lo) & (conf < hi)
            cal[mask] = self.bin_vals[i]
        return cal, pred


class SoftLabelTS(nn.Module):
    """
    Soft-Label Temperature Scaling (SLTS).

    Minimises cross-entropy with soft label targets:
        L = − Σ_k π_k(x) log softmax(z/T)_k
    Equivalent to KL(π(x) ‖ softmax(z/T)) up to an entropy constant.
    """

    def __init__(self, init_T: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels_soft: torch.Tensor) -> "SoftLabelTS":
        """
        Parameters
        ----------
        logits      : (N, K)
        labels_soft : (N, K) annotator probability distribution (rows sum to 1)
        """
        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                  tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            opt.zero_grad()
            log_p = torch.log_softmax(self(logits), dim=1)
            loss  = -(labels_soft * log_p).sum(1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @property
    def T(self) -> float:
        return self.temperature.item()


class MonteCarloTS(nn.Module):
    """
    Monte Carlo Temperature Scaling (MCTS).

    For each calibration example, draws `n_samples` labels from the empirical
    annotator distribution π̂(x) and averages their NLL.  This is the direct
    calibration analogue of Monte Carlo Conformal Prediction [Stutz et al. 2023].

    When the soft label distribution is exactly the empirical annotation
    frequency, MCTS with n_samples→∞ is equivalent to SLTS.
    """

    def __init__(self, init_T: float = 1.5, n_samples: int = 50, seed: int = 42):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_T)
        self.n_samples   = n_samples
        self.seed        = seed

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels_soft: torch.Tensor) -> "MonteCarloTS":
        """
        Parameters
        ----------
        logits      : (N, K)
        labels_soft : (N, K) annotator probability distribution
        """
        rng = np.random.default_rng(self.seed)
        N, K = labels_soft.shape
        lsoft_np = labels_soft.numpy()

        # Sample individual annotations
        rows, cols = [], []
        for i in range(N):
            p = lsoft_np[i]
            p = np.clip(p, 0, None)
            p = p / p.sum()          # ensure valid probability
            sampled = rng.choice(K, size=self.n_samples, p=p)
            rows.extend([i] * self.n_samples)
            cols.extend(sampled.tolist())

        idx_n  = torch.tensor(rows, dtype=torch.long)
        labels_flat = torch.tensor(cols, dtype=torch.long)
        logits_exp  = logits[idx_n]          # (N*n_samples, K)

        opt  = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                   tolerance_grad=1e-9, tolerance_change=1e-11)
        crit = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = crit(self(logits_exp), labels_flat)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @property
    def T(self) -> float:
        return self.temperature.item()


class VectorScaling(nn.Module):
    """
    Vector Scaling: one temperature parameter per class, soft-label variant.

    logits_calibrated_k = logit_k / T_k
    """

    def __init__(self, n_classes: int, init_T: float = 1.5):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(n_classes) * init_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperatures.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels_soft: torch.Tensor,
            lr: float = 0.05, n_epochs: int = 2000) -> "VectorScaling":
        opt = torch.optim.Adam([self.temperatures], lr=lr)

        for _ in range(n_epochs):
            opt.zero_grad()
            log_p = torch.log_softmax(self(logits), dim=1)
            loss  = -(labels_soft * log_p).sum(1).mean()
            loss.backward()
            opt.step()

        return self


# ──────────────────────────────────────────────────────────────────────────────
# Non-parametric methods  (operate on numpy arrays of predicted probabilities)
# ──────────────────────────────────────────────────────────────────────────────

class SoftHistogramBinning:
    """
    Histogram Binning with soft calibration targets.

    In each confidence bin, the calibration target is the mean annotator
    probability for the predicted class, rather than hard-label accuracy.
    The calibrated output is the bin-specific soft-accuracy target.
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins    = n_bins
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.bin_vals  = np.zeros(n_bins)   # calibrated target per bin

    def fit(self, probs: np.ndarray, labels_soft: np.ndarray) -> "SoftHistogramBinning":
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        sacc = labels_soft[np.arange(len(pred)), pred]

        for i, (lo, hi) in enumerate(zip(self.bin_edges[:-1], self.bin_edges[1:])):
            mask = (conf >= lo) & (conf < hi)
            self.bin_vals[i] = sacc[mask].mean() if mask.sum() > 0 else (lo + hi) / 2.0
        return self

    def calibrate(self, probs: np.ndarray):
        """Returns (calibrated_conf, predicted_class) arrays."""
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        cal  = np.zeros_like(conf)
        for i, (lo, hi) in enumerate(zip(self.bin_edges[:-1], self.bin_edges[1:])):
            mask = (conf >= lo) & (conf < hi)
            cal[mask] = self.bin_vals[i]
        return cal, pred


class SoftIsotonicRegression:
    """
    Isotonic Regression with soft calibration targets.

    Fits a monotone mapping confidence → mean soft accuracy using PAVA.
    """

    def __init__(self):
        self._ir = SkIso(out_of_bounds="clip", increasing=True)

    def fit(self, probs: np.ndarray, labels_soft: np.ndarray) -> "SoftIsotonicRegression":
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        sacc = labels_soft[np.arange(len(pred)), pred]
        self._ir.fit(conf, sacc)
        return self

    def calibrate(self, probs: np.ndarray):
        """Returns (calibrated_conf, predicted_class) arrays."""
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        return self._ir.predict(conf), pred


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: apply parametric calibrators to numpy arrays
# ──────────────────────────────────────────────────────────────────────────────

def apply_parametric(calibrator: nn.Module, logits_np: np.ndarray) -> np.ndarray:
    """Run a fitted torch calibrator on numpy logits and return numpy probs."""
    with torch.no_grad():
        t = torch.tensor(logits_np, dtype=torch.float32)
        p = torch.softmax(calibrator(t), dim=1)
    return p.numpy()
