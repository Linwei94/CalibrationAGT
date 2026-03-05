"""
Monte Carlo Temperature Scaling (MCTS) — Design & Analysis

Design
------
MCTS is the calibration analogue of Monte Carlo Conformal Prediction (MCCP)
from Stutz et al. (2023).

Given soft labels π̂(x_i) for each calibration example i:

  Step 1 (Sampling):   For s = 1…S, draw  ŷ_{i,s} ~ Categorical(π̂(x_i))
  Step 2 (Augmentation): Build dataset  D_aug = { (z_i, ŷ_{i,s}) | i,s }
                          of size N · S
  Step 3 (Fitting):    T* = argmin_T   (1/NS) Σ_{i,s} -log p(ŷ_{i,s}|z_i,T)

Key theoretical properties
--------------------------
  P1 (Consistency with SLTS):
       E_{ŷ~π̂(x)}[-log p(ŷ|z,T)] = -Σ_k π̂_k(x) log softmax(z/T)_k
       ⟹ MCTS loss (averaged over MC samples) equals SLTS loss.
       As S→∞, MCTS T* → SLTS T* (LLN).

  P2 (S=1 degenerates toward TS):
       With S=1, MCTS selects one annotation per example uniformly from π̂.
       If π̂ is one-hot (unambiguous), MCTS = TS.
       For ambiguous examples, expected label ≠ majority class ⟹ T*(MCTS,S=1)
       is still biased toward SLTS relative to TS.

  P3 (Variance decreases as 1/√S):
       Var[T*_MCTS] ∝ 1/S  (standard MC variance reduction).

Experiments (toy example, no GPU needed, ~60 s)
-----------------------------------------------
  1. S sensitivity: ECE-Soft and T* vs. S ∈ {1,2,3,5,8,12,20,30,50,100,200}
                    averaged over 10 seeds (shows convergence to SLTS)
  2. Reliability diagrams: TS / MCTS(S=5) / MCTS(S=20) / MCTS(S=50) / SLTS
  3. Variance analysis: T* distribution across seeds for S=1,5,50

Usage
-----
    cd experiments
    python run_mcts_analysis.py
    # Saves figures to ../paper/figs/
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

OUT = Path(__file__).parent.parent / "paper" / "figs"
OUT.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.labelsize":  10,
    "axes.titlesize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi":      180,
    "pdf.fonttype":    42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PAL = dict(
    ts="#E07040", slts="#4CAF80", mcts="#9B59B6",
    raw="#5A8FC2", gap="#C0392B",
    gap_over="#FFB3B3", gap_under="#C8F0C8",
)


# ══════════════════════════════════════════════════════════════════════════════
# Toy data and model
# ══════════════════════════════════════════════════════════════════════════════

def generate_data(n_per_class: int = 1500, seed: int = 0):
    """3-class toy: class 1 is ambiguous (π=[0, 0.7, 0.3])."""
    rng = np.random.default_rng(seed)
    cov = np.diag([1.0, 0.5])
    centers = [np.array([-3., 0.]), np.array([0., 0.]), np.array([3., 0.])]
    PI_AMB  = [0., 0.7, 0.3]

    X_list, yh_list, ys_list, amb_list = [], [], [], []
    for i, c in enumerate(centers):
        Xi = rng.multivariate_normal(c, cov, n_per_class).astype(np.float32)
        if i == 0:
            yh = np.zeros(n_per_class, dtype=np.int64)
            ys = np.tile([1., 0., 0.], (n_per_class, 1)).astype(np.float32)
            am = np.zeros(n_per_class, dtype=bool)
        elif i == 1:
            yh = np.ones(n_per_class, dtype=np.int64)
            ys = np.tile(PI_AMB, (n_per_class, 1)).astype(np.float32)
            am = np.ones(n_per_class, dtype=bool)
        else:
            yh = np.full(n_per_class, 2, dtype=np.int64)
            ys = np.tile([0., 0., 1.], (n_per_class, 1)).astype(np.float32)
            am = np.zeros(n_per_class, dtype=bool)
        X_list.append(Xi); yh_list.append(yh)
        ys_list.append(ys); amb_list.append(am)

    return (np.vstack(X_list), np.concatenate(yh_list),
            np.vstack(ys_list).astype(np.float32), np.concatenate(amb_list))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )
    def forward(self, x): return self.net(x)


# ── Calibration methods (compact, self-contained) ─────────────────────────────

class TS(nn.Module):
    """Standard Temperature Scaling: min_T NLL(softmax(z/T), hard_label)."""
    def __init__(self):
        super().__init__()
        self._T = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, z): return z / self._T.clamp(min=1e-3)

    def fit(self, z, yh):
        opt  = torch.optim.LBFGS([self._T], lr=0.1, max_iter=500,
                                   tolerance_grad=1e-9, tolerance_change=1e-11)
        crit = nn.CrossEntropyLoss()
        def cl(): opt.zero_grad(); l = crit(self(z), yh); l.backward(); return l
        opt.step(cl)
        return self

    @property
    def T(self) -> float: return self._T.item()


class SLTS(nn.Module):
    """
    Soft-Label Temperature Scaling: min_T KL(π̂ ‖ softmax(z/T)).

    This is the S→∞ limit of MCTS.
    """
    def __init__(self):
        super().__init__()
        self._T = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, z): return z / self._T.clamp(min=1e-3)

    def fit(self, z, ys):
        opt = torch.optim.LBFGS([self._T], lr=0.1, max_iter=500,
                                  tolerance_grad=1e-9, tolerance_change=1e-11)
        def cl():
            opt.zero_grad()
            loss = -(ys * torch.log_softmax(self(z), 1)).sum(1).mean()
            loss.backward()
            return loss
        opt.step(cl)
        return self

    @property
    def T(self) -> float: return self._T.item()


class MCTS(nn.Module):
    """
    Monte Carlo Temperature Scaling.

    Algorithm
    ---------
    1. For each calibration example i:
         Sample ŷ_{i,1}, …, ŷ_{i,S} ~ Categorical(π̂(xᵢ))
    2. Build augmented dataset:
         D_aug = { (zᵢ, ŷ_{i,s}) | i = 1…N, s = 1…S }
    3. Fit temperature:
         T* = argmin_T  (1/NS) Σ_{i,s} CE(softmax(zᵢ/T), ŷ_{i,s})

    Theorem (consistency): E_{ŷ}[CE(p(·|z,T), ŷ)] = KL(π̂ ‖ p(·|z,T)) + H(π̂)
    Hence MCTS minimises the same objective as SLTS in expectation, so T*_MCTS
    → T*_SLTS as S → ∞.

    Parameters
    ----------
    S    : Monte Carlo samples per example (↑S → lower variance, closer to SLTS)
    seed : RNG seed for reproducibility of the Monte Carlo sampling
    """

    def __init__(self, S: int = 50, seed: int = 42):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.S    = S
        self.seed = seed

    def forward(self, z):
        return z / self.temperature.clamp(min=1e-3)

    def fit(self, z: torch.Tensor, ys: torch.Tensor) -> "MCTS":
        """
        Parameters
        ----------
        z  : (N, K) logits on calibration set
        ys : (N, K) soft label distributions (rows sum to 1)
        """
        rng    = np.random.default_rng(self.seed)
        N, K   = ys.shape
        ys_np  = np.clip(ys.numpy(), 0, None)
        ys_np /= ys_np.sum(axis=1, keepdims=True)   # normalise (safety)

        # Step 1 & 2: sample and build augmented dataset
        rows, cols = [], []
        for i in range(N):
            sampled = rng.choice(K, size=self.S, p=ys_np[i])
            rows.extend([i] * self.S)
            cols.extend(sampled.tolist())

        z_aug   = z[torch.tensor(rows, dtype=torch.long)]    # (N*S, K)
        y_aug   = torch.tensor(cols, dtype=torch.long)        # (N*S,)

        # Step 3: fit T with LBFGS
        opt  = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=500,
                                   tolerance_grad=1e-9, tolerance_change=1e-11)
        crit = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = crit(self(z_aug), y_aug)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @property
    def T(self): return self.temperature.item()


# ── Metrics ───────────────────────────────────────────────────────────────────

def ece_soft(probs: np.ndarray, targets: np.ndarray,
             n_bins: int = 12, min_count: int = 3) -> tuple[float, list]:
    """ECE with soft targets. Returns (ece_value, bin_info_list)."""
    K    = probs.shape[1]
    soft = np.eye(K)[targets.astype(int)] if targets.ndim == 1 else targets.astype(float)
    conf = probs.max(1); pred = probs.argmax(1)
    sacc = soft[np.arange(len(pred)), pred]
    edges = np.linspace(0, 1, n_bins + 1)
    val, info = 0., []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        n = m.sum()
        if n >= min_count:
            mc, ma = float(conf[m].mean()), float(sacc[m].mean())
            val += (n / len(probs)) * abs(mc - ma)
            info.append((mc, ma, int(n)))
    return float(val), info


def apply_cal(cal, logits_np: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(cal(torch.tensor(logits_np, dtype=torch.float32)), 1).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Setup: toy data, train model, establish baselines
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  Monte Carlo Temperature Scaling — Design & Analysis")
print("=" * 65)
print("\n[1/4] Data generation and model training …")

np.random.seed(42); torch.manual_seed(42)
X, y_hard, y_soft, amb = generate_data(n_per_class=1500)

idx = np.arange(len(X))
idx_tr, idx_rest = train_test_split(idx, test_size=.40, random_state=0)
idx_cal, idx_te  = train_test_split(idx_rest, test_size=.50, random_state=0)

Xtr_t  = torch.FloatTensor(X[idx_tr])
ytr_t  = torch.LongTensor(y_hard[idx_tr])
Xcl_t  = torch.FloatTensor(X[idx_cal])
ycl_t  = torch.LongTensor(y_hard[idx_cal])
yscl_t = torch.FloatTensor(y_soft[idx_cal])
Xte_t  = torch.FloatTensor(X[idx_te])

model = MLP()
opt_m = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit  = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=128, shuffle=True)
for _ in range(250):
    model.train()
    for xb, yb in loader:
        opt_m.zero_grad(); crit(model(xb), yb).backward(); opt_m.step()

model.eval()
with torch.no_grad():
    lg_cal = model(Xcl_t)
    lg_te  = model(Xte_t)
lg_te_np = lg_te.numpy()

yh_te  = y_hard[idx_te]
ys_te  = y_soft[idx_te]
amb_te = amb[idx_te]
yh1hot = np.eye(3)[yh_te]

# Baseline calibrators
ts_m   = TS().fit(lg_cal, ycl_t)
slts_m = SLTS().fit(lg_cal, yscl_t)

p_raw  = torch.softmax(lg_te, 1).numpy()
p_ts   = apply_cal(ts_m,   lg_te_np)
p_slts = apply_cal(slts_m, lg_te_np)

ece_raw,  _ = ece_soft(p_raw,  ys_te)
ece_ts,   _ = ece_soft(p_ts,   ys_te)
ece_slts, _ = ece_soft(p_slts, ys_te)

print(f"  Uncalibrated : ECE-Soft = {ece_raw *100:.2f}%   T = —")
print(f"  TS           : ECE-Soft = {ece_ts  *100:.2f}%   T = {ts_m.T:.3f}  "
      f"{'← T<1: wrong direction!' if ts_m.T < 1 else ''}")
print(f"  SLTS (S→∞)   : ECE-Soft = {ece_slts*100:.2f}%   T = {slts_m.T:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: S sensitivity — ECE-Soft and T* vs S
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/4] MCTS sensitivity to S (N_seeds=10 per S) …")

S_values = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 100, 200]
N_SEEDS  = 10

# Results containers
ece_by_S  = {S: [] for S in S_values}   # ECE-Soft per seed
T_by_S    = {S: [] for S in S_values}   # T* per seed
ece_h_by_S = {S: [] for S in S_values}  # ECE-Hard per seed (vs hard labels)

for S in S_values:
    for seed in range(N_SEEDS):
        m = MCTS(S=S, seed=seed).fit(lg_cal, yscl_t)
        p = apply_cal(m, lg_te_np)
        e_s, _ = ece_soft(p, ys_te)
        e_h, _ = ece_soft(p, yh1hot)
        ece_by_S[S].append(e_s * 100)
        T_by_S[S].append(m.T)
        ece_h_by_S[S].append(e_h * 100)

    mu_e = np.mean(ece_by_S[S])
    sd_e = np.std(ece_by_S[S])
    mu_T = np.mean(T_by_S[S])
    sd_T = np.std(T_by_S[S])
    print(f"  S={S:4d}: ECE-Soft={mu_e:.2f}±{sd_e:.2f}%   T*={mu_T:.3f}±{sd_T:.3f}")

print(f"\n  SLTS reference : ECE-Soft={ece_slts*100:.2f}%  T*={slts_m.T:.3f}")
print(f"  TS   reference : ECE-Soft={ece_ts  *100:.2f}%  T*={ts_m.T:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Fixed S reliability diagrams
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3/4] Reliability diagrams for selected S …")
S_selected = [1, 5, 20, 50]
mcts_selected = {}
for S in S_selected:
    m = MCTS(S=S, seed=42).fit(lg_cal, yscl_t)
    p = apply_cal(m, lg_te_np)
    e, _ = ece_soft(p, ys_te)
    mcts_selected[S] = {"probs": p, "T": m.T, "ece": e * 100}
    print(f"  S={S:3d}: ECE-Soft={e*100:.2f}%  T*={m.T:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4/4] Generating figures …")

# ── Helper: reliability diagram panel ─────────────────────────────────────────

def rel_panel(ax, probs, targets, title, color, n_bins=12):
    e, info = ece_soft(probs, targets, n_bins=n_bins)
    if not info:
        ax.set_title(f"{title}\nECE={e*100:.2f}%"); return e
    confs = np.array([b[0] for b in info])
    accs  = np.array([b[1] for b in info])
    w = 0.9 / n_bins
    for c, a in zip(confs, accs):
        col = PAL["gap_under"] if a > c else PAL["gap_over"]
        ax.fill_between([c-w/2, c+w/2], [min(c,a)]*2, [max(c,a)]*2,
                        color=col, alpha=0.5, zorder=1)
    ax.bar(confs, accs, width=w*0.80, color=color, alpha=0.85, zorder=2)
    ax.plot([0,1],[0,1], "--", color="#444", lw=1.3, zorder=3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Avg. label prob.")
    ax.set_title(f"{title}\nECE-Soft = {e*100:.2f}%", fontweight="bold")
    ax.grid(True, alpha=0.20)
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Figure A: MCTS Convergence Analysis (3 panels)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle(
    "Monte Carlo Temperature Scaling — Design Analysis\n"
    "As $S\\to\\infty$, MCTS T* and ECE-Soft converge to the SLTS (closed-form) solution",
    fontweight="bold", fontsize=11,
)

emu = np.array([np.mean(ece_by_S[S]) for S in S_values])
esd = np.array([np.std(ece_by_S[S])  for S in S_values])
Tmu = np.array([np.mean(T_by_S[S])   for S in S_values])
Tsd = np.array([np.std(T_by_S[S])    for S in S_values])

# Panel 1: ECE-Soft vs S
ax = axes[0]
ax.fill_between(S_values, emu - esd, emu + esd, alpha=0.25, color=PAL["mcts"])
ax.plot(S_values, emu, "o-", color=PAL["mcts"], lw=2.0, ms=5,
        label=f"MCTS  (mean ± 1σ, {N_SEEDS} seeds)")
ax.axhline(ece_slts * 100, color=PAL["slts"], lw=2.0, ls="--",
           label=f"SLTS  (S→∞):  {ece_slts*100:.2f}%")
ax.axhline(ece_ts   * 100, color=PAL["ts"],   lw=2.0, ls=":",
           label=f"TS    (hard-label):  {ece_ts*100:.2f}%")
ax.axhline(ece_raw  * 100, color=PAL["raw"],  lw=1.5, ls="-.", alpha=0.7,
           label=f"Uncalibrated:  {ece_raw*100:.2f}%")
ax.set_xscale("log")
ax.set_xlabel("$S$ (Monte Carlo samples per example)")
ax.set_ylabel("ECE-Soft (%)")
ax.set_title("Convergence of ECE-Soft to SLTS", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

# Panel 2: T* vs S
ax = axes[1]
ax.fill_between(S_values, Tmu - Tsd, Tmu + Tsd, alpha=0.25, color=PAL["mcts"])
ax.plot(S_values, Tmu, "o-", color=PAL["mcts"], lw=2.0, ms=5, label="MCTS T* (mean ± 1σ)")
ax.axhline(slts_m.T, color=PAL["slts"], lw=2.0, ls="--",
           label=f"SLTS T* = {slts_m.T:.3f}")
ax.axhline(ts_m.T,   color=PAL["ts"],   lw=2.0, ls=":",
           label=f"TS T* = {ts_m.T:.3f}")
ax.axhline(1.0, color="gray", lw=1.2, ls="-", alpha=0.6, label="T=1 (no calibration)")
ax.set_xscale("log")
ax.set_xlabel("$S$ (Monte Carlo samples per example)")
ax.set_ylabel("Optimal temperature $T^*$")
ax.set_title("Convergence of $T^*$ to SLTS", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

# Panel 3: Variance (std of T*) vs S  — should scale as 1/√S
ax = axes[2]
ax.plot(S_values, Tsd, "o-", color=PAL["mcts"], lw=2.0, ms=5, label="σ(T*) empirical")
# Fit a 1/sqrt(S) reference line
S_ref = np.array(S_values, dtype=float)
ref_scale = Tsd[0] * np.sqrt(S_values[0])           # scale to match S=1 point
ax.plot(S_values, ref_scale / np.sqrt(S_ref), "--",
        color="gray", lw=1.8, label=r"$\propto 1/\sqrt{S}$ (MC variance)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("$S$")
ax.set_ylabel("σ($T^*$) — standard deviation over seeds")
ax.set_title(r"Variance Reduction: $\sigma(T^*) \propto 1/\sqrt{S}$", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25, which="both")

plt.tight_layout()
plt.savefig(OUT / "fig_mcts_convergence.pdf", bbox_inches="tight")
plt.close()
print("  → fig_mcts_convergence.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure B: Reliability Diagrams — TS | MCTS(S=1,5,20,50) | SLTS
# ─────────────────────────────────────────────────────────────────────────────

n_panels = 2 + len(S_selected)   # TS + selected S's + SLTS
fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 2.8, 4.0),
                         constrained_layout=True)
fig.suptitle(
    "Reliability Diagrams (Soft Evaluation): TS | MCTS at Various S | SLTS\n"
    "Each panel evaluated against the annotator distribution π̂(x)",
    fontweight="bold", fontsize=10,
)

rel_panel(axes[0], p_ts, ys_te,
          f"TS  (hard labels)\nT*={ts_m.T:.3f}", PAL["ts"])

for j, S in enumerate(S_selected):
    d = mcts_selected[S]
    rel_panel(axes[1 + j], d["probs"], ys_te,
              f"MCTS  S={S}\nT*={d['T']:.3f}", PAL["mcts"])

rel_panel(axes[-1], p_slts, ys_te,
          f"SLTS  (S→∞)\nT*={slts_m.T:.3f}", PAL["slts"])

axes[-1].legend(handles=[
    mpatches.Patch(color=PAL["gap_over"],  alpha=0.7, label="Overconfident"),
    mpatches.Patch(color=PAL["gap_under"], alpha=0.7, label="Underconfident"),
], loc="upper left", fontsize=7.5)

plt.savefig(OUT / "fig_mcts_reliability.pdf", bbox_inches="tight")
plt.close()
print("  → fig_mcts_reliability.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure C: Combined summary (for paper integration)
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38,
                        height_ratios=[1, 1])
fig.suptitle(
    "Monte Carlo Temperature Scaling (MCTS) — Full Analysis\n"
    r"Toy example: 3-class MLP, class 1 ambiguous ($\hat{\pi}=[0, 0.7, 0.3]$)",
    fontweight="bold", fontsize=11,
)

# Row 0: convergence panels
ax_ece = fig.add_subplot(gs[0, 0])
ax_T   = fig.add_subplot(gs[0, 1])
ax_var = fig.add_subplot(gs[0, 2])

# ECE-Soft vs S
ax_ece.fill_between(S_values, emu - esd, emu + esd, alpha=0.22, color=PAL["mcts"])
ax_ece.plot(S_values, emu, "o-", color=PAL["mcts"], lw=2, ms=4.5, label="MCTS")
ax_ece.axhline(ece_slts * 100, color=PAL["slts"], lw=2, ls="--",
               label=f"SLTS: {ece_slts*100:.2f}%")
ax_ece.axhline(ece_ts   * 100, color=PAL["ts"],   lw=2, ls=":",
               label=f"TS: {ece_ts*100:.2f}%")
ax_ece.set_xscale("log")
ax_ece.set_xlabel("$S$"); ax_ece.set_ylabel("ECE-Soft (%)")
ax_ece.set_title("ECE-Soft vs. $S$", fontweight="bold")
ax_ece.legend(fontsize=8); ax_ece.grid(True, alpha=0.25)

# T* vs S
ax_T.fill_between(S_values, Tmu - Tsd, Tmu + Tsd, alpha=0.22, color=PAL["mcts"])
ax_T.plot(S_values, Tmu, "o-", color=PAL["mcts"], lw=2, ms=4.5, label="MCTS $T^*$")
ax_T.axhline(slts_m.T, color=PAL["slts"], lw=2, ls="--",
             label=f"SLTS: {slts_m.T:.3f}")
ax_T.axhline(ts_m.T,   color=PAL["ts"],   lw=2, ls=":",
             label=f"TS: {ts_m.T:.3f}")
ax_T.axhline(1.0, color="gray", lw=1.1, ls="-", alpha=0.5, label="T=1")
ax_T.set_xscale("log")
ax_T.set_xlabel("$S$"); ax_T.set_ylabel("$T^*$")
ax_T.set_title("Temperature $T^*$ vs. $S$", fontweight="bold")
ax_T.legend(fontsize=8); ax_T.grid(True, alpha=0.25)

# Variance vs S
ax_var.plot(S_values, Tsd, "o-", color=PAL["mcts"], lw=2, ms=4.5, label=r"$\sigma(T^*)$ empirical")
ax_var.plot(S_values, ref_scale / np.sqrt(S_ref), "--", color="gray", lw=1.8,
            label=r"$\propto 1/\sqrt{S}$")
ax_var.set_xscale("log"); ax_var.set_yscale("log")
ax_var.set_xlabel("$S$"); ax_var.set_ylabel(r"$\sigma(T^*)$")
ax_var.set_title(r"Variance Reduction ($\propto 1/\sqrt{S}$)", fontweight="bold")
ax_var.legend(fontsize=8); ax_var.grid(True, alpha=0.25, which="both")

# Row 1: reliability diagrams for TS / MCTS(S=5) / MCTS(S=50) / SLTS
panels_r1 = [
    ("TS (hard labels)",   p_ts,                        PAL["ts"],   ts_m.T,   False),
    (f"MCTS  S=5",         mcts_selected[5]["probs"],   PAL["mcts"], mcts_selected[5]["T"],  False),
    (f"MCTS  S=50",        mcts_selected[50]["probs"],  PAL["mcts"], mcts_selected[50]["T"], False),
    ("SLTS  (S→∞)",        p_slts,                      PAL["slts"], slts_m.T, True),
]

for col, (title, probs, color, T_val, show_legend) in enumerate(panels_r1):
    ax = fig.add_subplot(gs[1, col] if col < 3 else gs[1, 2])
    rel_panel(ax, probs, ys_te, title, color)
    ax.text(0.97, 0.04, f"$T^*={T_val:.3f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.85))
    if show_legend:
        ax.legend(handles=[
            mpatches.Patch(color=PAL["gap_over"],  alpha=0.7, label="Over-conf."),
            mpatches.Patch(color=PAL["gap_under"], alpha=0.7, label="Under-conf."),
        ], loc="upper left", fontsize=7.5)

plt.savefig(OUT / "fig_mcts_full.pdf", bbox_inches="tight")
plt.close()
print("  → fig_mcts_full.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Print summary table
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  Results Summary")
print("="*65)
print(f"  {'Method':<22} {'ECE-Soft':>10}  {'T*':>8}")
print(f"  {'-'*42}")
print(f"  {'Uncalibrated':<22} {ece_raw*100:>9.2f}%  {'—':>8}")
print(f"  {'TS (hard labels)':<22} {ece_ts*100:>9.2f}%  {ts_m.T:>8.3f}")
for S in [1, 5, 20, 50, 100, 200]:
    mu_e = np.mean(ece_by_S[S])
    sd_e = np.std(ece_by_S[S])
    mu_T = np.mean(T_by_S[S])
    sd_T = np.std(T_by_S[S])
    print(f"  {'MCTS (S='+str(S)+')':<22} {mu_e:>7.2f}±{sd_e:.2f}%  {mu_T:>6.3f}±{sd_T:.3f}")
print(f"  {'SLTS (S→∞)':<22} {ece_slts*100:>9.2f}%  {slts_m.T:>8.3f}")
print("="*65)
print(f"\nAll figures saved to:  {OUT}")
print("  fig_mcts_convergence.pdf  — 3-panel convergence analysis")
print("  fig_mcts_reliability.pdf  — reliability diagram comparison")
print("  fig_mcts_full.pdf         — combined (6-panel paper figure)")
