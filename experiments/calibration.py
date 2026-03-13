"""
Calibration methods for the ambiguous ground truth setting.

Methods
-------
  TemperatureScaling  (TS)    — NLL vs. hard (voted) labels          [Guo et al. 2017]
  PlattScaling        (PS)    — matrix scaling (W,b) vs. hard labels [Platt 1999 / Kull 2019]
  SoftLabelTS         (SLTS)  — KL(π(x) ‖ softmax(z/T))             [ours]
  MonteCarloTS        (MCTS)  — sample individual annotations → NLL  [ours, cf. Stutz 2023]
  VectorScaling       (VS)    — per-class temperatures, ambiguity-aware targets [ours]
  PseudoSoftLabelTS   (PSLTS) — annotation-free; ε_i=1-f_i[y*]      [ours]
  LabelSmoothTS       (LS-TS) — global label smoothing baseline      [ours]
  HardHistogramBinning(HB-H)  — non-parametric, hard accuracy targets[Zadrozny & Elkan 2001]
  SoftHistogramBinning(HB-S)  — non-parametric, distributional targets [ours]
  SoftIsotonicReg     (IR)    — isotonic regression, distributional targets [ours]
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


class SoftPlattScaling(nn.Module):
    """
    Soft-Label Platt / Matrix Scaling.

    Same diagonal affine transform as PlattScaling (W·z + b with diagonal W),
    but trained against the annotator distribution using KL divergence.
    This is the natural ambiguity-aware extension of PlattScaling, and isolates
    whether gains come from (a) soft targets alone or (b) parametric form.
    """

    def __init__(self, n_classes: int, lr: float = 0.01, n_epochs: int = 2000):
        super().__init__()
        self.W = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))
        self._lr      = lr
        self._n_epochs = n_epochs

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.W + self.b

    def fit(self, logits: torch.Tensor, labels_soft: torch.Tensor) -> "SoftPlattScaling":
        opt = torch.optim.Adam([self.W, self.b], lr=self._lr, weight_decay=1e-4)

        for _ in range(self._n_epochs):
            opt.zero_grad()
            log_p = torch.log_softmax(self(logits), dim=1)
            loss  = -(labels_soft * log_p).sum(1).mean()
            loss.backward()
            opt.step()

        return self


class DirichletCalibration(nn.Module):
    """
    Dirichlet Calibration [Kull et al. NeurIPS 2019].

    Learns a full affine transformation W·z + b of the logit vector, where W is a
    full K×K matrix (not just diagonal as in Platt/Matrix Scaling).  More expressive
    than Temperature Scaling or Platt Scaling.

    fit_hard  — calibrate against hard (voted) labels  [Kull et al. baseline]
    fit_soft  — calibrate against soft label distributions  [our extension]

    Regularisation: L2 on off-diagonal entries of W to prevent overfitting on small
    calibration sets (Kull et al. recommend mu=1e-3 for the ODIR variant).
    """

    def __init__(self, n_classes: int, lr: float = 0.01, n_epochs: int = 2000,
                 l2: float = 1e-3):
        super().__init__()
        self.W = nn.Parameter(torch.eye(n_classes))   # K×K weight matrix
        self.b = nn.Parameter(torch.zeros(n_classes))  # K-dim bias
        self._lr      = lr
        self._n_epochs = n_epochs
        self._l2       = l2

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits @ self.W.T + self.b

    def _fit(self, logits: torch.Tensor, targets: torch.Tensor,
             soft: bool) -> "DirichletCalibration":
        opt = torch.optim.Adam([self.W, self.b], lr=self._lr, weight_decay=1e-4)

        for _ in range(self._n_epochs):
            opt.zero_grad()
            scaled = self(logits)
            if soft:
                log_p = torch.log_softmax(scaled, dim=1)
                loss  = -(targets * log_p).sum(1).mean()
            else:
                loss  = nn.CrossEntropyLoss()(scaled, targets)
            # Off-diagonal regularisation (ODIR)
            K = self.W.shape[0]
            off_diag = self.W * (1 - torch.eye(K, device=self.W.device))
            loss = loss + self._l2 * (off_diag ** 2).sum()
            loss.backward()
            opt.step()

        return self

    def fit_hard(self, logits: torch.Tensor,
                 labels_hard: torch.Tensor) -> "DirichletCalibration":
        return self._fit(logits, labels_hard, soft=False)

    def fit_soft(self, logits: torch.Tensor,
                 labels_soft: torch.Tensor) -> "DirichletCalibration":
        return self._fit(logits, labels_soft, soft=True)


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


class PseudoSoftLabelTS(nn.Module):
    """
    Pseudo-Soft Label Temperature Scaling (PSLTS).

    Annotation-free calibration that targets ECE_true using only voted labels.
    Constructs per-example pseudo soft labels from model confidence:

        ε_i  = 1 − f_i[y*]         (disagreement proxy: 1 minus voted-class confidence)
        π̂_i = f_i[y*]·e_{y*} + (1−f_i[y*])·(1/K)   (smooth toward uniform)

    Then optimises T via KL(π̂_i ‖ softmax(z_i/T)).

    Provable: T*_PSLTS > T*_TS  (pseudo-label is softer than one-hot)
    Annotation-free: uses only model logits and voted labels.
    """

    def __init__(self, init_T: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    @staticmethod
    def make_pseudo_labels(logits: torch.Tensor,
                           labels_hard: torch.Tensor) -> torch.Tensor:
        """
        Build pseudo soft labels from model logits and voted labels.

        Parameters
        ----------
        logits      : (N, K)
        labels_hard : (N,)  integer class labels

        Returns
        -------
        pi_hat : (N, K)  pseudo soft label distribution
        """
        with torch.no_grad():
            K   = logits.shape[1]
            f   = torch.softmax(logits, dim=1)                    # (N, K)
            conf = f[torch.arange(len(labels_hard)), labels_hard] # (N,) f_i[y*]
            yh  = torch.zeros_like(f)
            yh.scatter_(1, labels_hard.unsqueeze(1), 1.0)         # one-hot
            eps = (1.0 - conf).clamp(0, 1)                        # ε_i in [0,1]
            pi_hat = (1 - eps).unsqueeze(1) * yh + (eps / K).unsqueeze(1)
        return pi_hat

    def fit(self, logits: torch.Tensor,
            labels_hard: torch.Tensor) -> "PseudoSoftLabelTS":
        """
        Parameters
        ----------
        logits      : (N, K)
        labels_hard : (N,)  integer voted class labels
        """
        pi_hat = self.make_pseudo_labels(logits, labels_hard)

        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                  tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            opt.zero_grad()
            log_p = torch.log_softmax(self(logits), dim=1)
            loss  = -(pi_hat * log_p).sum(1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @property
    def T(self) -> float:
        return self.temperature.item()


class LabelSmoothTS(nn.Module):
    """
    Label-Smooth Temperature Scaling (LS-TS).

    Non-adaptive baseline: uses a global ε = mean(1 − f_i[y*]) for all examples.

        π̂_i = (1−ε)·e_{y*} + ε/K

    Serves as an ablation for PSLTS: same average smoothing but non-adaptive.
    """

    def __init__(self, init_T: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    @staticmethod
    def make_pseudo_labels(logits: torch.Tensor,
                           labels_hard: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            K    = logits.shape[1]
            f    = torch.softmax(logits, dim=1)
            conf = f[torch.arange(len(labels_hard)), labels_hard]
            eps  = (1.0 - conf).mean().item()                     # global ε
            yh   = torch.zeros_like(f)
            yh.scatter_(1, labels_hard.unsqueeze(1), 1.0)
            pi_hat = (1 - eps) * yh + (eps / K)
        return pi_hat

    def fit(self, logits: torch.Tensor,
            labels_hard: torch.Tensor) -> "LabelSmoothTS":
        pi_hat = self.make_pseudo_labels(logits, labels_hard)

        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                  tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            opt.zero_grad()
            log_p = torch.log_softmax(self(logits), dim=1)
            loss  = -(pi_hat * log_p).sum(1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @property
    def T(self) -> float:
        return self.temperature.item()


class FixedLabelSmoothTS(nn.Module):
    """
    Fixed Label-Smooth TS (Fixed-LS-TS): ablation baseline for LS-TS.

    Uses a fixed, data-independent smoothing weight ε (default 0.1) identical
    for all examples, without any adaptation from the model or calibration data.
    This tests whether LS-TS's improvement is due to soft targets in general
    (any ε > 0) versus the data-driven estimate of ε.
    """

    def __init__(self, eps: float = 0.1, init_T: float = 1.5):
        super().__init__()
        self.eps = eps
        self.temperature = nn.Parameter(torch.ones(1) * init_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor,
            labels_hard: torch.Tensor) -> "FixedLabelSmoothTS":
        with torch.no_grad():
            K  = logits.shape[1]
            yh = torch.zeros(len(labels_hard), K)
            yh.scatter_(1, labels_hard.unsqueeze(1), 1.0)
            pi_hat = (1 - self.eps) * yh + (self.eps / K)

        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                  tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            opt.zero_grad()
            log_p = torch.log_softmax(self(logits), dim=1)
            loss  = -(pi_hat * log_p).sum(1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @property
    def T(self) -> float:
        return self.temperature.item()


class EntropyLabelSmoothTS(nn.Module):
    """
    Entropy-based Label-Smooth TS (Ent-LS-TS): ablation baseline for LS-TS.

    Uses per-instance smoothing weight ε_i = H(f(x_i)) / log(K), where H is
    the Shannon entropy of the model's softmax output.  This tests whether
    per-instance entropy from the model's own predictions is a better proxy
    for annotation ambiguity than the global mean-confidence estimate in LS-TS.
    """

    def __init__(self, init_T: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    @staticmethod
    def make_pseudo_labels(logits: torch.Tensor,
                           labels_hard: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            K  = logits.shape[1]
            f  = torch.softmax(logits, dim=1)
            H  = -(f * f.clamp(min=1e-12).log()).sum(1)  # Shannon entropy (N,)
            eps_i = (H / np.log(K)).clamp(0, 1)           # normalised to [0,1]
            yh = torch.zeros_like(f)
            yh.scatter_(1, labels_hard.unsqueeze(1), 1.0)
            pi_hat = (1 - eps_i).unsqueeze(1) * yh + (eps_i / K).unsqueeze(1)
        return pi_hat

    def fit(self, logits: torch.Tensor,
            labels_hard: torch.Tensor) -> "EntropyLabelSmoothTS":
        pi_hat = self.make_pseudo_labels(logits, labels_hard)

        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                  tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            opt.zero_grad()
            log_p = torch.log_softmax(self(logits), dim=1)
            loss  = -(pi_hat * log_p).sum(1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @property
    def T(self) -> float:
        return self.temperature.item()


class EMSmoothTS(nn.Module):
    """
    EM Label-Smooth Temperature Scaling (EM-LS-TS).

    Iterative version of LS-TS: repeats the E-step (re-estimate ε from
    calibrated softmax) and M-step (re-optimise T) for multiple rounds.

    Round 0 is identical to standard LS-TS.
    """

    def __init__(self, init_T: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_T)
        self.history: list[dict] = []   # stores per-round {eps, T}

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels_hard: torch.Tensor,
            n_rounds: int = 1) -> "EMSmoothTS":
        K = logits.shape[1]
        self.history = []

        for rnd in range(n_rounds):
            # E-step: estimate ε from current calibrated softmax
            with torch.no_grad():
                f = torch.softmax(self(logits), dim=1)
                conf = f[torch.arange(len(labels_hard)), labels_hard]
                eps = (1.0 - conf).mean().item()
                yh = torch.zeros_like(f)
                yh.scatter_(1, labels_hard.unsqueeze(1), 1.0)
                pi_hat = (1 - eps) * yh + (eps / K)

            # M-step: optimise T given pseudo-labels
            self.temperature = nn.Parameter(torch.ones(1) * max(self.temperature.item(), 0.5))
            opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                      tolerance_grad=1e-9, tolerance_change=1e-11)

            def closure():
                opt.zero_grad()
                log_p = torch.log_softmax(self(logits), dim=1)
                loss = -(pi_hat * log_p).sum(1).mean()
                loss.backward()
                return loss

            opt.step(closure)
            self.history.append({"round": rnd, "eps": eps, "T": self.temperature.item()})

        return self

    @property
    def T(self) -> float:
        return self.temperature.item()


class ClassCondLabelSmoothTS(nn.Module):
    """
    Class-Conditional Label-Smooth TS (CC-LS-TS).

    Uses per-class smoothing: ε_k = mean_{i: y*_i=k}(1 - f_i[y*_i]).
    Captures the fact that some classes are more ambiguous than others.
    """

    def __init__(self, init_T: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_T)
        self.eps_per_class: list[float] = []

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor,
            labels_hard: torch.Tensor) -> "ClassCondLabelSmoothTS":
        with torch.no_grad():
            K = logits.shape[1]
            f = torch.softmax(logits, dim=1)
            conf = f[torch.arange(len(labels_hard)), labels_hard]

            # Per-class ε
            eps_k = torch.zeros(K)
            for k in range(K):
                mask = labels_hard == k
                if mask.sum() > 0:
                    eps_k[k] = (1.0 - conf[mask]).mean()
            self.eps_per_class = eps_k.tolist()

            # Build pseudo-labels with per-class ε
            eps_i = eps_k[labels_hard]  # (N,)
            yh = torch.zeros_like(f)
            yh.scatter_(1, labels_hard.unsqueeze(1), 1.0)
            pi_hat = (1 - eps_i).unsqueeze(1) * yh + (eps_i / K).unsqueeze(1)

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
    def T(self) -> float:
        return self.temperature.item()


class AdaptiveTempScaling(nn.Module):
    """
    Adaptive Temperature Scaling (ATS).

    Learns a per-instance temperature T(z_i) from four logit-derived features:
        φ(z) = [max_logit, H(softmax(z))/log K, top1−top2 prob gap, max_prob]
    T(z) = softplus(w·φ(z) + b) + 0.1   (linear map, 5 parameters total)

    Annotation-free: trained with voted-label NLL, like TS.
    At test time, T is predicted from each example's own logits.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.4)  # softplus(0.4)+0.1 ≈ 1.1 init

    @staticmethod
    def _features(logits: torch.Tensor) -> torch.Tensor:
        """Compute 4 scalar features per example; result detached from logit grad."""
        with torch.no_grad():
            K     = logits.shape[1]
            p     = torch.softmax(logits, dim=1)
            s, _  = p.sort(dim=1, descending=True)
            feat  = torch.stack([
                logits.max(dim=1).values,                    # max logit
                -(p * (p + 1e-9).log()).sum(1) / np.log(K), # norm entropy
                s[:, 0] - s[:, 1],                           # top1-top2 gap
                s[:, 0],                                     # max prob
            ], dim=1)
        return feat.detach()

    def get_temperatures(self, logits: torch.Tensor) -> torch.Tensor:
        """Return per-instance temperatures (N,)."""
        feat = self._features(logits)
        return torch.nn.functional.softplus(self.linear(feat).squeeze(1)) + 0.1

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = self.get_temperatures(logits)
        return logits / T.unsqueeze(1).clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels_hard: torch.Tensor,
            l2: float = 1e-3, n_epochs: int = 500) -> "AdaptiveTempScaling":
        feat = self._features(logits)
        crit = nn.CrossEntropyLoss()
        opt  = torch.optim.Adam(self.parameters(), lr=5e-3)
        for _ in range(n_epochs):
            opt.zero_grad()
            T   = torch.nn.functional.softplus(self.linear(feat).squeeze(1)) + 0.1
            loss = crit(logits / T.unsqueeze(1), labels_hard)
            loss += l2 * (self.linear.weight ** 2).sum()
            loss.backward()
            opt.step()
        return self

    @property
    def T(self) -> float:
        return float("nan")  # per-instance; no single global T

    def mean_T(self, logits: torch.Tensor) -> float:
        return self.get_temperatures(logits).mean().item()


class AdaptiveSoftLabelTS(nn.Module):
    """
    Adaptive Soft-Label Temperature Scaling (ATS-Soft).

    Same per-instance T(z_i) architecture as AdaptiveTempScaling but trained
    with KL divergence against soft annotator targets — requires annotator data.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.4)

    def get_temperatures(self, logits: torch.Tensor) -> torch.Tensor:
        feat = AdaptiveTempScaling._features(logits)
        return torch.nn.functional.softplus(self.linear(feat).squeeze(1)) + 0.1

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = self.get_temperatures(logits)
        return logits / T.unsqueeze(1).clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels_soft: torch.Tensor,
            l2: float = 1e-3, n_epochs: int = 500) -> "AdaptiveSoftLabelTS":
        feat = AdaptiveTempScaling._features(logits)
        opt  = torch.optim.Adam(self.parameters(), lr=5e-3)
        for _ in range(n_epochs):
            opt.zero_grad()
            T     = torch.nn.functional.softplus(self.linear(feat).squeeze(1)) + 0.1
            log_p = torch.log_softmax(logits / T.unsqueeze(1), dim=1)
            loss  = -(labels_soft * log_p).sum(1).mean()
            loss += l2 * (self.linear.weight ** 2).sum()
            loss.backward()
            opt.step()
        return self

    @property
    def T(self) -> float:
        return float("nan")

    def mean_T(self, logits: torch.Tensor) -> float:
        return self.get_temperatures(logits).mean().item()


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
        lsoft_np = labels_soft.detach().cpu().numpy()

        # Sample individual annotations
        rows, cols = [], []
        for i in range(N):
            p = lsoft_np[i]
            p = np.clip(p, 0, None)
            p = p / p.sum()          # ensure valid probability
            sampled = rng.choice(K, size=self.n_samples, p=p)
            rows.extend([i] * self.n_samples)
            cols.extend(sampled.tolist())

        device = logits.device
        idx_n  = torch.tensor(rows, dtype=torch.long, device=device)
        labels_flat = torch.tensor(cols, dtype=torch.long, device=device)
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
    Vector Scaling: one temperature parameter per class, ambiguity-aware variant.

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
