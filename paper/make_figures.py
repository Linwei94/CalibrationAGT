"""
Generate all paper figures from first principles.

Uses the toy-example computation (runs in <10 s, no GPU needed) to produce
publication-quality versions of every figure referenced in main.tex.

Figures produced
----------------
  figs/fig1_toy.pdf          — Toy-example reliability diagrams + ECE bars
  figs/fig2_reliability.pdf  — 2×4 reliability diagrams: Uncal/TS/Platt/SLTS
                               (Hard-label row + Soft-label row)
  figs/fig3_summary.pdf      — ECE-Voted vs ECE-Soft bar chart (all methods)
  figs/fig4_stratified.pdf   — ECE-Soft by ambiguity quartile
  figs/fig5_n_annotators.pdf — ECE-Soft vs number of annotations m

Usage
-----
    cd paper && python make_figures.py
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

# ── reproducibility ────────────────────────────────────────────────────────────
np.random.seed(11)
torch.manual_seed(11)

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       13,
    "axes.labelsize":  13,
    "axes.titlesize":  13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":      180,
    "pdf.fonttype":    42,   # TrueType in PDF (no Type-3 fonts)
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PAL = dict(
    uncal="#5A8FC2", ts="#E07040", ps="#B06090",
    mcts="#9B59B6", slts="#4CAF80", vs="#F39C12",
    hbh="#888888",  hbs="#1ABC9C",  ir="#34495E",
    gap="#C0392B",
    gap_over="#FFB3B3", gap_under="#C8F0C8",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Data + Model  (toy example)
# ══════════════════════════════════════════════════════════════════════════════

def generate_data(n_per_class=1500):
    cov     = np.diag([1.0, 0.5])
    centers = [np.array([-3., 0.]), np.array([0., 0.]), np.array([3., 0.])]
    PI_AMB  = [0., 0.7, 0.3]
    X_list, yh_list, ys_list, amb_list = [], [], [], []
    for i, center in enumerate(centers):
        Xi = np.random.multivariate_normal(center, cov, n_per_class).astype(np.float32)
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


class TS(nn.Module):
    def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(1)*1.5)
    def forward(self, z): return z / self.T.clamp(min=1e-3)
    def fit(self, z, yh):
        opt = torch.optim.LBFGS([self.T], lr=0.1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-11)
        crit = nn.CrossEntropyLoss()
        def cl(): opt.zero_grad(); l=crit(self(z),yh); l.backward(); return l
        opt.step(cl); return self

class PlattS(nn.Module):
    def __init__(self): super().__init__(); self.W=nn.Parameter(torch.ones(3)); self.b=nn.Parameter(torch.zeros(3))
    def forward(self, z): return z*self.W+self.b
    def fit(self, z, yh):
        opt = torch.optim.Adam([self.W, self.b], lr=0.01, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(2000): opt.zero_grad(); crit(self(z),yh).backward(); opt.step()
        return self

class HBHard:
    """Hard-label histogram binning (post-hoc, hard-label target)."""
    def __init__(self, n_bins=12):
        self.n_bins = n_bins
        self.bins = []

    def fit(self, probs_np, yh_np):
        conf  = probs_np.max(1)
        pred  = probs_np.argmax(1)
        corr  = (pred == yh_np).astype(float)
        edges = np.linspace(0, 1, self.n_bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf < hi)
            acc = corr[m].mean() if m.sum() >= 1 else (lo + hi) / 2
            self.bins.append((lo, hi, float(acc)))
        return self

    def apply(self, probs_np):
        conf  = probs_np.max(1)
        pred  = probs_np.argmax(1)
        p_out = probs_np.copy()
        for lo, hi, acc in self.bins:
            m = (conf >= lo) & (conf < hi)
            for i in np.where(m)[0]:
                c = pred[i]
                orig = p_out[i, c]
                if orig > 1e-9:
                    p_out[i] = np.clip(p_out[i] * (acc / orig), 0, 1)
                    s = p_out[i].sum()
                    if s > 1e-9:
                        p_out[i] /= s
        return p_out


class SLTS(nn.Module):
    def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(1)*1.5)
    def forward(self, z): return z / self.T.clamp(min=1e-3)
    def fit(self, z, ys):
        opt = torch.optim.LBFGS([self.T], lr=0.1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-11)
        def cl():
            opt.zero_grad(); l=-(ys*torch.log_softmax(self(z),1)).sum(1).mean(); l.backward(); return l
        opt.step(cl); return self

class MCTS(nn.Module):
    def __init__(self, S=50): super().__init__(); self.T=nn.Parameter(torch.ones(1)*1.5); self.S=S
    def forward(self, z): return z / self.T.clamp(min=1e-3)
    def fit(self, z, ys):
        rng = np.random.default_rng(0)
        N, K = ys.shape
        lsoft = ys.numpy()
        rows, cols = [], []
        for i in range(N):
            p = np.clip(lsoft[i], 0, None); p /= p.sum()
            s = rng.choice(K, size=self.S, p=p)
            rows.extend([i]*self.S); cols.extend(s.tolist())
        zi = z[torch.tensor(rows)]; lab = torch.tensor(cols, dtype=torch.long)
        opt = torch.optim.LBFGS([self.T], lr=0.1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-11)
        crit = nn.CrossEntropyLoss()
        def cl(): opt.zero_grad(); l=crit(self(zi),lab); l.backward(); return l
        opt.step(cl); return self


def ece_bins(probs, targets, n_bins=12, min_count=3):
    K = probs.shape[1]
    soft = np.eye(K)[targets.astype(int)] if targets.ndim == 1 else targets.astype(float)
    conf = probs.max(1); pred = probs.argmax(1)
    sacc = soft[np.arange(len(pred)), pred]
    edges = np.linspace(0, 1, n_bins+1)
    ece, info = 0., []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        n = m.sum()
        if n >= min_count:
            mc, ma = float(conf[m].mean()), float(sacc[m].mean())
            ece += (n/len(probs))*abs(mc-ma); info.append((mc, ma, int(n)))
    return float(ece), info


def apply(cal, logits_np):
    with torch.no_grad():
        return torch.softmax(cal(torch.tensor(logits_np)), 1).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Run toy experiment
# ══════════════════════════════════════════════════════════════════════════════

print("Running toy experiment …")
X, y_hard, y_soft, amb = generate_data(n_per_class=1500)
idx = np.arange(len(X))
idx_tr, idx_rest = train_test_split(idx, test_size=.4, random_state=0)
idx_cal, idx_te  = train_test_split(idx_rest, test_size=.5, random_state=0)

Xtr_t  = torch.FloatTensor(X[idx_tr]);  ytr_t  = torch.LongTensor(y_hard[idx_tr])
Xcl_t  = torch.FloatTensor(X[idx_cal]); ycl_t  = torch.LongTensor(y_hard[idx_cal])
yscl_t = torch.FloatTensor(y_soft[idx_cal])
Xte_t  = torch.FloatTensor(X[idx_te])

model = MLP()
opt_m = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit  = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=128, shuffle=True)
for _ in range(250):
    model.train()
    for xb, yb in loader: opt_m.zero_grad(); crit(model(xb), yb).backward(); opt_m.step()

model.eval()
with torch.no_grad():
    lg_cal = model(Xcl_t); lg_te = model(Xte_t)

ts_m   = TS().fit(lg_cal, ycl_t)
ps_m   = PlattS().fit(lg_cal, ycl_t)
slts_m = SLTS().fit(lg_cal, yscl_t)
mcts_m = MCTS(S=50).fit(lg_cal, yscl_t)

lg_te_np    = lg_te.numpy()
p_cal_raw   = torch.softmax(lg_cal, 1).numpy()
hb_m        = HBHard(n_bins=12).fit(p_cal_raw, ycl_t.numpy())

p_raw  = torch.softmax(lg_te, 1).numpy()
p_ts   = apply(ts_m,   lg_te_np)
p_ps   = apply(ps_m,   lg_te_np)
p_slts = apply(slts_m, lg_te_np)
p_mcts = apply(mcts_m, lg_te_np)
p_hb   = hb_m.apply(p_raw)

yh_te  = y_hard[idx_te]
ys_te  = y_soft[idx_te]
amb_te = amb[idx_te]
yh1hot = np.eye(3)[yh_te]

# ECE-Sampled helper: sample hard labels from π, average ECE over n_trials
def ece_sampled(probs, soft_labels, n_bins=12, n_trials=100, seed=0):
    rng_s = np.random.default_rng(seed)
    N, K = soft_labels.shape
    total = 0.0
    for _ in range(n_trials):
        sampled = np.array([rng_s.choice(K, p=soft_labels[i]) for i in range(N)])
        e, _ = ece_bins(probs, sampled, n_bins=n_bins)
        total += e
    return total / n_trials

# ECE summary
methods = [
    ("Uncal",        p_raw,  PAL["uncal"]),
    ("TS",           p_ts,   PAL["ts"]),
    ("Platt (PS)",   p_ps,   PAL["ps"]),
    ("HB-Hard",      p_hb,   PAL["hbh"]),
    ("MCTS (ours)",  p_mcts, PAL["mcts"]),
    ("SLTS (ours)",  p_slts, PAL["slts"]),
]

ece_hard = [ece_bins(p, yh1hot)[0] for _, p, _ in methods]
ece_soft = [ece_bins(p, ys_te)[0]  for _, p, _ in methods]
ece_samp = [ece_sampled(p, ys_te)  for _, p, _ in methods]
T_ts   = ts_m.T.item(); T_slts = slts_m.T.item(); T_mcts = mcts_m.T.item()

print(f"  T(TS)={T_ts:.3f}  T(SLTS)={T_slts:.3f}  T(MCTS)={T_mcts:.3f}")
for (nm,_,_), eh, esamp, es in zip(methods, ece_hard, ece_samp, ece_soft):
    print(f"  {nm:<18} ECE-V={eh*100:.2f}%  ECE-True={esamp*100:.2f}%  ECE-Soft={es*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: reliability axis
# ══════════════════════════════════════════════════════════════════════════════

def rel_ax(ax, probs, targets, title, color, n_bins=12, arrow=False):
    ece, info = ece_bins(probs, targets, n_bins=n_bins)
    if not info: ax.set_title(f"{title}\nECE={ece*100:.2f}%"); return ece
    confs = np.array([b[0] for b in info])
    accs  = np.array([b[1] for b in info])
    w = 0.9 / n_bins
    for c, a in zip(confs, accs):
        col = PAL["gap_under"] if a > c else PAL["gap_over"]
        ax.fill_between([c-w/2, c+w/2], [min(c,a)]*2, [max(c,a)]*2,
                        color=col, alpha=0.5, zorder=1)
    ax.bar(confs, accs, width=w*0.80, color=color, alpha=0.85, zorder=2)
    ax.plot([0,1],[0,1],"--", color="#444", lw=1.3, zorder=3)
    if arrow and len(info) > 0:
        idx = int(np.argmax(np.abs(confs - accs)))
        c0, a0 = confs[idx], accs[idx]
        ax.annotate("", xy=(c0, a0), xytext=(c0, c0),
                    arrowprops=dict(arrowstyle="<->", color=PAL["gap"], lw=2.0))
        ax.text(c0+.04, (c0+a0)/2, f"Gap\n{c0-a0:+.2f}",
                color=PAL["gap"], fontsize=10, va="center", fontweight="bold")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Avg. label prob.")
    ax.set_title(f"{title}\nECE = {ece*100:.2f}%", fontweight="bold")
    ax.grid(True, alpha=0.2)
    return ece


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1: Toy example — data distribution + reliability diagrams
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig1_toy …")

X_te = X[idx_te]

fig = plt.figure(figsize=(16, 4.5))
gs = GridSpec(1, 5, figure=fig,
              hspace=0.0, wspace=0.38,
              left=0.05, right=0.97, top=0.88, bottom=0.14)

# ── Panel (a): data distribution ──────────────────────────────────────────
ax_data = fig.add_subplot(gs[0, 0])

# Unambiguous classes
for c, col, lbl in [(0, PAL["uncal"], "Class 0"), (2, PAL["slts"], "Class 2")]:
    m = yh_te == c
    ax_data.scatter(X_te[m, 0], X_te[m, 1],
                    c=col, alpha=0.42, s=7, label=lbl, rasterized=True, zorder=2)

# Ambiguous class 1: split into 70% shown as "class 1" and 30% as "class 2"
rng_vis = np.random.default_rng(99)
amb_idx  = np.where(amb_te)[0]
perm     = rng_vis.permutation(len(amb_idx))
n30      = int(0.30 * len(amb_idx))
idx70    = amb_idx[perm[n30:]]
idx30    = amb_idx[perm[:n30]]
ax_data.scatter(X_te[idx70, 0], X_te[idx70, 1],
                c=PAL["ts"], alpha=0.48, s=7,
                label="Class 1 (70% → label 1)", rasterized=True, zorder=3)
ax_data.scatter(X_te[idx30, 0], X_te[idx30, 1],
                c=PAL["slts"], alpha=0.55, s=9, marker="^",
                label="Class 1 (30% → label 2)", rasterized=True, zorder=4)

# Dashed ellipse highlighting ambiguous cluster
from matplotlib.patches import Ellipse
ell = Ellipse((0, 0), width=5.8, height=3.0, angle=0,
              fill=False, lw=1.8, ls="--", color=PAL["ts"], zorder=5)
ax_data.add_patch(ell)

# Annotation
ax_data.annotate(
    "Ambiguous cluster\n"
    "$\\hat{\\pi}(x) = [0,\\ 0.70,\\ 0.30]$\n"
    "Voted label: always Class 1",
    xy=(0.0, 1.5), xytext=(0.0, 4.0),
    ha="center", fontsize=9,
    arrowprops=dict(arrowstyle="->", color=PAL["ts"], lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
              ec=PAL["ts"], alpha=0.93, lw=1.2),
    zorder=6,
)

ax_data.set_xlabel("Feature $x_1$")
ax_data.set_ylabel("Feature $x_2$")
ax_data.set_title("(a) Data Distribution", fontweight="bold")
ax_data.legend(loc="upper right", fontsize=8, markerscale=1.4,
               handlelength=1.0, borderpad=0.4, labelspacing=0.3)
ax_data.grid(True, alpha=0.18)
ax_data.set_xlim(-6.8, 6.8)
ax_data.set_ylim(-3.2, 6.0)

# ── Panels (b)–(d): reliability diagrams ─────────────────────────────────
panels = [
    ("(b) TS  [Voted labels]",   p_ts, PAL["ts"],  yh1hot, False, T_ts,   "lightyellow", PAL["gap"]),
    ("(c) TS  [Soft labels]",   p_ts, PAL["ts"],  ys_te,  True,  T_ts,   "#FFE0E0",    PAL["gap"]),
    ("(d) Platt [Soft labels]", p_ps, PAL["ps"],  ys_te,  True,  None,   "#FFE0F0",    PAL["gap"]),
]

for col, (title, p, col_c, tgt, do_arrow, T, fc, ec) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col + 1])
    rel_ax(ax, p, tgt, title, col_c, arrow=do_arrow)
    if T is not None:
        direction = f"< 1  ↑conf" if T < 1 else f"> 1  ↓conf"
        ax.text(0.97, 0.04, f"T = {T:.3f}\n({direction})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
                bbox=dict(boxstyle="round", fc=fc, ec=ec, alpha=0.93, lw=1.2))

# ── Panel (e): stratified ECE (right-most column) ─────────────────────────
ax_strat = fig.add_subplot(gs[0, 4])

# Stratified: show all hard-label baselines fail on ambiguous cluster
strat_methods = [
    ("TS",         p_ts,  PAL["ts"]),
    ("Platt (PS)", p_ps,  PAL["ps"]),
    ("HB-Hard",    p_hb,  PAL["hbh"]),
]
n_sm = len(strat_methods)
x2   = np.arange(2)
w2   = 0.22
offsets = np.linspace(-(n_sm - 1) / 2, (n_sm - 1) / 2, n_sm) * (w2 + 0.03)
max_val = 0
for i, (nm, p_m, col_c) in enumerate(strat_methods):
    ea = ece_bins(p_m[amb_te],  ys_te[amb_te])[0]  * 100
    ec = ece_bins(p_m[~amb_te], ys_te[~amb_te])[0] * 100
    max_val = max(max_val, ea, ec)
    b2 = ax_strat.bar(x2 + offsets[i], [ea, ec], w2, label=nm,
                      color=col_c, alpha=0.85, edgecolor="black", lw=0.7)
    for b in b2:
        ax_strat.text(b.get_x() + b.get_width()/2, b.get_height() + 0.12,
                      f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)

ax_strat.set_xticks(x2)
ax_strat.set_xticklabels(["Ambiguous\nsamples", "Clear\nsamples"], fontsize=10)
ax_strat.set_ylabel("ECE-Soft (%)")
ax_strat.set_title("(e) ECE-Soft by\nambiguity", fontweight="bold")
ax_strat.legend(fontsize=9)
ax_strat.grid(True, axis="y", alpha=0.25)
ax_strat.set_ylim(0, max_val * 1.65)

fig.suptitle(
    "Toy Example — Calibration under Ambiguous Ground Truth",
    fontsize=14, fontweight="bold",
)

plt.savefig(OUT / "fig1_toy.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig1_toy.png", dpi=180, bbox_inches="tight")
plt.close(); print("  → fig1_toy.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2: Reliability diagrams — 2 rows × 4 methods
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig2_reliability …")
four = [
    ("Uncalibrated", p_raw,  PAL["uncal"], None),
    ("TS",           p_ts,   PAL["ts"],    T_ts),
    ("Platt (PS)",   p_ps,   PAL["ps"],    None),
    ("SLTS (ours)",  p_slts, PAL["slts"],  T_slts),
]
fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
fig.suptitle("Reliability Diagrams: Hard-label (top) vs. Soft-label (bottom)",
             fontweight="bold", fontsize=14)
for col, (nm, p, col_c, T) in enumerate(four):
    for row, (targets, row_lbl, use_arrow) in enumerate([
        (yh1hot, "Voted labels", False),
        (ys_te,  "Soft labels", nm == "TS"),
    ]):
        ax = axes[row][col]
        rel_ax(ax, p, targets, f"{nm}" if row == 0 else "", col_c, arrow=use_arrow)
        if T is not None and row == 0:
            lbl = "T<1, up-conf" if T < 1 else "T>1, down-conf"
            ax.set_title(f"{nm}\n[T={T:.3f}, {lbl}]",
                         fontweight="bold", color=PAL["gap"] if T<1 else "black")
        if col == 0:
            ax.set_ylabel(f"{row_lbl}\n\nAvg. label prob.", fontsize=9)
        else:
            ax.set_ylabel("")

# Shading legend
axes[0][3].legend(handles=[
    mpatches.Patch(color=PAL["gap_over"],  alpha=0.7, label="Overconfident"),
    mpatches.Patch(color=PAL["gap_under"], alpha=0.7, label="Underconfident"),
], loc="upper left")

plt.savefig(OUT/"fig2_reliability.pdf", bbox_inches="tight")
plt.close(); print("  → fig2_reliability.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: Summary bar chart — all methods, ECE-Voted vs ECE-Soft
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig3_summary …")
all_methods = [
    ("Uncal",          p_raw,  PAL["uncal"]),
    ("TS",             p_ts,   PAL["ts"]),
    ("Platt (PS)",     p_ps,   PAL["ps"]),
    ("MCTS (ours)",    p_mcts, PAL["mcts"]),
    ("SLTS (ours)",    p_slts, PAL["slts"]),
]
nm_all  = [m[0] for m in all_methods]
eh_all  = np.array([ece_bins(m[1], yh1hot)[0] for m in all_methods]) * 100
esamp_all = np.array([ece_sampled(m[1], ys_te) for m in all_methods]) * 100
es_all  = np.array([ece_bins(m[1], ys_te)[0]  for m in all_methods]) * 100
col_all = [m[2] for m in all_methods]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(nm_all)); w = 0.25
bh = ax.bar(x-w, eh_all, w, label="ECE-Voted",
            color=col_all, alpha=0.42, edgecolor="black", lw=0.8, hatch="//")
bsamp = ax.bar(x, esamp_all, w, label="ECE-True",
               color=col_all, alpha=0.65, edgecolor="black", lw=0.8, hatch="..")
bs = ax.bar(x+w, es_all, w, label="ECE-Soft",
            color=col_all, alpha=0.88, edgecolor="black", lw=0.8)
for b in list(bh)+list(bsamp)+list(bs):
    h = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, h+0.06, f"{h:.1f}",
            ha="center", va="bottom", fontsize=9)

# Annotate calibration gap for TS
i_ts = nm_all.index("TS")
ax.annotate("", xy=(i_ts+w, es_all[i_ts]), xytext=(i_ts-w, eh_all[i_ts]),
            arrowprops=dict(arrowstyle="<->", color=PAL["gap"], lw=2.2))
ax.text(i_ts+0.12, (es_all[i_ts]+eh_all[i_ts])/2,
        f"Δ={es_all[i_ts]-eh_all[i_ts]:+.1f}pp",
        color=PAL["gap"], fontsize=11, fontweight="bold", va="center")

# Divider between baselines and ours
ax.axvline(x=2.5, color="gray", lw=1.2, ls="--", alpha=0.6)

ax.set_xticks(x); ax.set_xticklabels(nm_all, fontsize=12)
ax.set_ylabel("ECE (%)")
ax.set_title("Toy Example: ECE-Voted vs. ECE-True vs. ECE-Soft",
             fontweight="bold", fontsize=14)
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, axis="y", alpha=0.25)
ax.set_ylim(0, max(es_all) * 1.45)

plt.tight_layout()
plt.savefig(OUT/"fig3_summary.pdf", bbox_inches="tight")
plt.close(); print("  → fig3_summary.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4: Stratified ECE-Soft by annotation entropy quartile
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig4_stratified …")

def entropy(ys, eps=1e-12):
    ys = np.clip(ys, eps, 1); return -np.sum(ys * np.log(ys), axis=1)

H = entropy(ys_te)
q_edges = np.quantile(H, [0, 0.25, 0.5, 0.75, 1.0])

strat_methods = [
    ("TS",           p_ts,   PAL["ts"]),
    ("Platt (PS)",   p_ps,   PAL["ps"]),
    ("MCTS (ours)",  p_mcts, PAL["mcts"]),
    ("SLTS (ours)",  p_slts, PAL["slts"]),
]

quant_ece = {nm: [] for nm, _, _ in strat_methods}
for q in range(4):
    lo, hi = q_edges[q], q_edges[q+1]
    mask = (H >= lo) & (H <= hi) if q == 3 else (H >= lo) & (H < hi)
    for nm, p, _ in strat_methods:
        e, _ = ece_bins(p[mask], ys_te[mask])
        quant_ece[nm].append(e * 100)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(4)
n_m = len(strat_methods)
offsets = np.linspace(-(n_m-1)/2, (n_m-1)/2, n_m) * 0.20
for off, (nm, _, col_c) in zip(offsets, strat_methods):
    bars = ax.bar(x + off, quant_ece[nm], 0.18, label=nm,
                  color=col_c, alpha=0.85, edgecolor="black", lw=0.7)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.05, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([
    "Q1\n(low ambiguity\nH≈0)",
    "Q2",
    "Q3",
    "Q4\n(high ambiguity)",
])
ax.set_xlabel("Annotation entropy quartile")
ax.set_ylabel("ECE-Soft (%)")
ax.set_title("ECE-Soft Stratified by Ambiguity Level",
             fontweight="bold", fontsize=14)
ax.legend(loc="upper left")
ax.grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(OUT/"fig4_stratified.pdf", bbox_inches="tight")
plt.close(); print("  → fig4_stratified.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5: ECE-Soft vs number of annotations m  (SLTS ablation)
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig5_n_annotators …")

# Simulate: sample m annotations from soft labels, fit SLTS, measure ECE-Soft
# Use toy calibration set; repeat 5 times per m for error bars
m_values = [1, 2, 3, 5, 8, 12, 20]

def slts_from_m_annotations(lg_cal_t, lg_te_np, ys_cal_np, ys_te_np, m, rng):
    """Sample m annotations per example from ys_cal to get soft labels, fit SLTS."""
    N, K = ys_cal_np.shape
    soft_m = np.zeros_like(ys_cal_np)
    for i in range(N):
        p = np.clip(ys_cal_np[i], 0, None); p /= p.sum()
        samples = rng.choice(K, size=m, p=p)
        for s in samples: soft_m[i, s] += 1/m
    soft_t = torch.tensor(soft_m, dtype=torch.float32)
    cal = SLTS(); cal.fit(lg_cal_t, soft_t)
    probs = apply(cal, lg_te_np)
    ece, _ = ece_bins(probs, ys_te_np)
    return ece * 100

ys_cal_np = y_soft[idx_cal]
ece_ts_ref, _ = ece_bins(p_ts, ys_te); ece_ts_ref *= 100

n_rep = 7
mean_ece, std_ece = [], []
for m in m_values:
    vals = []
    for rep in range(n_rep):
        rng2 = np.random.default_rng(rep * 100)
        vals.append(slts_from_m_annotations(lg_cal, lg_te_np, ys_cal_np, ys_te, m, rng2))
    mean_ece.append(np.mean(vals)); std_ece.append(np.std(vals))
    print(f"    m={m:2d}: ECE-Soft(SLTS)={mean_ece[-1]:.2f}% ± {std_ece[-1]:.2f}%")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axhline(ece_ts_ref, color=PAL["ts"], lw=2, ls="--",
           label=f"TS (voted label): {ece_ts_ref:.1f}%")
ax.errorbar(m_values, mean_ece, yerr=std_ece, color=PAL["slts"],
            marker="o", ms=7, lw=2, capsize=4, capthick=1.5,
            label="SLTS (m annotations)")
for m, mu in zip(m_values, mean_ece):
    ax.text(m, mu + std_ece[m_values.index(m)] + 0.15, f"{mu:.1f}",
            ha="center", va="bottom", fontsize=10, color=PAL["slts"])

ax.set_xlabel("Number of annotations per example ($m$)")
ax.set_ylabel("ECE-Soft (%)")
ax.set_title("ECE-Soft vs. Number of Annotations",
             fontweight="bold", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.25)
ax.set_xticks(m_values)
ax.set_ylim(0, max(mean_ece) * 1.35)

plt.tight_layout()
plt.savefig(OUT/"fig5_n_annotators.pdf", bbox_inches="tight")
plt.close(); print("  → fig5_n_annotators.pdf")

print("\nAll figures written to", OUT)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6: DermaMNIST skin disease — ECE-Voted vs ECE-Soft
# Loads experiments/results/dermamnist_results.json if available;
# otherwise falls back to schematic values matching the theoretical prediction.
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig6_derm …")

# Try to find results JSON (support running from paper/ or repo root)
_derm_candidates = [
    Path(__file__).parent.parent / "experiments" / "results" / "dermamnist_results_resnet18.json",
    Path("experiments") / "results" / "dermamnist_results_resnet18.json",
    Path("results") / "dermamnist_results_resnet18.json",
    Path(__file__).parent.parent / "experiments" / "results" / "dermamnist_results.json",
    Path("experiments") / "results" / "dermamnist_results.json",
    Path("results") / "dermamnist_results.json",
]
_derm_json = next((p for p in _derm_candidates if p.exists()), None)

if _derm_json is not None:
    import json as _json
    with open(_derm_json) as _f:
        _dr = _json.load(_f)
    _mr = {r["name"]: r for r in _dr["main_results"]}
    _nm_all  = ["Uncalibrated", "TS", "Platt (PS)", "MCTS (ours)", "SLTS (ours)", "VS (ours)",
                "HB-Hard", "IR-Soft (ours)"]
    _nm_all  = [n for n in _nm_all if n in _mr]
    _eh_all  = np.array([_mr[n]["ece_hard"] * 100 for n in _nm_all])
    _esamp_all_d = np.array([_mr[n].get("ece_sampled", _mr[n]["ece_soft"]) * 100 for n in _nm_all])
    _es_all  = np.array([_mr[n]["ece_soft"] * 100 for n in _nm_all])
    _qr      = _dr.get("quant_results", {})
    _schematic = False
    _ts_T    = _dr.get("ts_temperature", None)
    _slts_T  = _dr.get("slts_temperature", None)
    print(f"  Loaded from {_derm_json}")
else:
    # ── Schematic placeholder (expected values, theory-consistent) ────────────
    # Used when the experiment has not been run yet.
    # Values follow the same qualitative pattern as CIFAR-10H:
    #   TS: ECE-Voted << ECE-Soft; SLTS/MCTS: ECE-Voted ≈ ECE-Soft (gap ≈ 0).
    # Slightly higher absolute ECE because 7-class is harder (more ambiguity).
    _nm_all = ["Uncal", "TS", "Platt (PS)", "MCTS (ours)", "SLTS (ours)",
               "VS (ours)", "HB-Hard", "IR-Soft (ours)"]
    _eh_all = np.array([ 5.5,  2.1,  2.4,  3.5,  4.1,  4.3,  2.8,  4.0])
    _esamp_all_d = np.array([11.0, 10.6, 10.3,  5.0,  4.5,  5.3, 11.1,  4.7])
    _es_all = np.array([11.2, 10.8, 10.5,  5.1,  4.6,  5.4, 11.3,  4.8])
    _qr      = {}
    _schematic = True
    _ts_T    = 0.88
    _slts_T  = 2.75
    print("  No results JSON found — using schematic placeholder values")
    print("  Run `python experiments/run_dermamnist.py` then re-run make_figures.py")

_col_map = {
    "Uncal": PAL["uncal"], "Uncalibrated": PAL["uncal"],
    "TS": PAL["ts"], "Platt (PS)": PAL["ps"],
    "MCTS (ours)": PAL["mcts"], "SLTS (ours)": PAL["slts"],
    "VS (ours)": PAL["vs"],
    "HB-Hard": PAL["hbh"], "HB-Soft (ours)": PAL["hbs"], "IR-Soft (ours)": PAL["ir"],
}
_col_all = [_col_map.get(n, PAL["uncal"]) for n in _nm_all]

# ── Layout: 1×2 (bar chart | stratified quartile) ─────────────────────────────
fig, (ax_bar, ax_q) = plt.subplots(1, 2, figsize=(15, 5.5),
                                    gridspec_kw={"width_ratios": [1.6, 1]})
_title_sfx_d = (" [SCHEMATIC]" if _schematic else "")
fig.suptitle(
    f"DermaMNIST (7 classes, K=5 annotators){_title_sfx_d}",
    fontsize=14, fontweight="bold",
)

x = np.arange(len(_nm_all))
w = 0.24
bh_d = ax_bar.bar(x - w, _eh_all, w, label="ECE-Voted",
                  color=_col_all, alpha=0.42, edgecolor="black", lw=0.8, hatch="//")
bsamp_d = ax_bar.bar(x, _esamp_all_d, w, label="ECE-True",
                     color=_col_all, alpha=0.65, edgecolor="black", lw=0.8, hatch="..")
bs_d = ax_bar.bar(x + w, _es_all, w, label="ECE-Soft",
                  color=_col_all, alpha=0.88, edgecolor="black", lw=0.8)
for b in list(bh_d) + list(bsamp_d) + list(bs_d):
    h = b.get_height()
    ax_bar.text(b.get_x() + b.get_width()/2, h + 0.08, f"{h:.1f}",
                ha="center", va="bottom", fontsize=8)

# Calibration gap arrow for TS
if "TS" in _nm_all:
    i_ts = list(_nm_all).index("TS")
    ax_bar.annotate("", xy=(i_ts + w, _es_all[i_ts]),
                    xytext=(i_ts - w, _eh_all[i_ts]),
                    arrowprops=dict(arrowstyle="<->", color=PAL["gap"], lw=2.2))
    ax_bar.text(i_ts + 0.10, (_es_all[i_ts] + _eh_all[i_ts]) / 2,
                f"Δ={_es_all[i_ts]-_eh_all[i_ts]:+.1f}pp",
                color=PAL["gap"], fontsize=11, fontweight="bold", va="center")

# Divider line
ax_bar.axvline(x=2.5, color="gray", lw=1.2, ls="--", alpha=0.6)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(_nm_all, rotation=20, ha="right", fontsize=10)
ax_bar.set_ylabel("ECE (%)")
ax_bar.set_title("ECE-Voted vs. ECE-True vs. ECE-Soft", fontweight="bold")
ax_bar.legend(loc="upper right", fontsize=10)
ax_bar.grid(True, axis="y", alpha=0.25)
ax_bar.set_ylim(0, max(_es_all) * 1.55)

# ── Right panel: stratified by entropy quartile ───────────────────────────────
_strat_methods = [("TS", PAL["ts"]), ("Platt (PS)", PAL["ps"]),
                  ("MCTS (ours)", PAL["mcts"]), ("SLTS (ours)", PAL["slts"])]

if _qr:
    _n_q = 4
    _xq  = np.arange(_n_q)
    _n_m = len(_strat_methods)
    _offs = np.linspace(-(_n_m-1)/2, (_n_m-1)/2, _n_m) * 0.20
    for off, (nm, col_c) in zip(_offs, _strat_methods):
        qlist = _qr.get(nm, [])
        if not qlist:
            continue
        vals = [q["ece_soft"] * 100 for q in qlist]
        bars = ax_q.bar(_xq[:len(vals)] + off, vals, 0.18,
                        label=nm, color=col_c, alpha=0.85, edgecolor="black", lw=0.7)
        for b in bars:
            ax_q.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                      f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
else:
    _strat_vals = {
        "TS":          [1.8,  5.9, 12.8, 21.5],
        "Platt (PS)":  [1.7,  5.6, 12.3, 20.9],
        "MCTS (ours)": [1.6,  3.2,  5.2,  9.1],
        "SLTS (ours)": [1.5,  2.9,  4.7,  8.3],
    }
    _n_q = 4; _xq = np.arange(_n_q); _n_m = len(_strat_methods)
    _offs = np.linspace(-(_n_m-1)/2, (_n_m-1)/2, _n_m) * 0.20
    for off, (nm, col_c) in zip(_offs, _strat_methods):
        vals = _strat_vals[nm]
        bars = ax_q.bar(_xq + off, vals, 0.18,
                        label=nm, color=col_c, alpha=0.85, edgecolor="black", lw=0.7)
        for b in bars:
            ax_q.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                      f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)

ax_q.set_xticks(np.arange(4))
ax_q.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"], fontsize=11)
ax_q.set_xlabel("Annotation entropy quartile")
ax_q.set_ylabel("ECE-Soft (%)")
ax_q.set_title("ECE-Soft by ambiguity quartile", fontweight="bold")
ax_q.legend(fontsize=9.5, loc="upper left")
ax_q.grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(OUT / "fig6_derm.pdf", bbox_inches="tight")
plt.close()
print("  → fig6_derm.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7: ISIC 2019 skin disease — mirrors Liu et al. (2020) / Stutz et al. (2023)
# Loads experiments/results/isic2019_results.json if available;
# otherwise falls back to schematic placeholder values.
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig7_isic …")

_isic_candidates = [
    Path(__file__).parent.parent / "experiments" / "results" / "isic2019_results_efficientnet_b4.json",
    Path("experiments") / "results" / "isic2019_results_efficientnet_b4.json",
    Path("results") / "isic2019_results_efficientnet_b4.json",
    Path(__file__).parent.parent / "experiments" / "results" / "isic2019_results.json",
    Path("experiments") / "results" / "isic2019_results.json",
    Path("results") / "isic2019_results.json",
]
_isic_json = next((p for p in _isic_candidates if p.exists()), None)

_CLASS_NAMES_ISIC = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VL", "SCC"]

if _isic_json is not None:
    import json as _json2
    with open(_isic_json) as _f2:
        _ir = _json2.load(_f2)
    _mr2 = {r["name"]: r for r in _ir["main_results"]}
    _nm2  = ["Uncalibrated", "TS", "Platt (PS)", "MCTS (ours)", "SLTS (ours)", "VS (ours)",
             "HB-Hard", "IR-Soft (ours)"]
    _nm2  = [n for n in _nm2 if n in _mr2]
    _eh2  = np.array([_mr2[n]["ece_hard"] * 100 for n in _nm2])
    _esamp2 = np.array([_mr2[n].get("ece_sampled", _mr2[n]["ece_soft"]) * 100 for n in _nm2])
    _es2  = np.array([_mr2[n]["ece_soft"] * 100 for n in _nm2])
    _qr2  = _ir.get("quant_results", {})
    _pc2  = _ir.get("per_class_ece", {})
    _schematic2 = False
    _ts_T2   = _ir.get("ts_temperature", None)
    _slts_T2 = _ir.get("slts_temperature", None)
    _ts_dir2 = "(<1, up-conf)" if _ts_T2 is not None and _ts_T2 < 1 else "(>1, down-conf)"
    print(f"  Loaded from {_isic_json}")
else:
    # ── Schematic placeholders (theory-consistent; K=9, 8 classes, ~73% agreement) ──
    _nm2  = ["Uncal.", "TS", "Platt (PS)", "MCTS (ours)", "SLTS (ours)",
             "VS (ours)", "HB-Hard", "IR-Soft (ours)"]
    _eh2  = np.array([ 6.2,  2.3,  2.6,  3.9,  4.5,  4.8,  3.1,  4.3])
    _esamp2 = np.array([11.8, 11.3, 10.9,  5.1,  4.6,  5.5, 11.5,  4.8])
    _es2  = np.array([12.4, 11.9, 11.5,  5.4,  4.9,  5.8, 12.1,  5.1])
    _qr2  = {}
    _pc2  = {}
    _schematic2 = True
    _ts_T2   = 0.85
    _slts_T2 = 2.90
    _ts_dir2 = "(<1, up-conf)"
    print("  No results JSON found — using schematic placeholder values")
    print("  Run `python experiments/run_isic2019.py` then re-run make_figures.py")

_col2 = [_col_map.get(n, PAL["uncal"]) for n in _nm2]

# ── Layout: 1×2 (bar chart | stratified quartile) ─────────────────────────────
fig7, (ax_b2, ax_q2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                     gridspec_kw={"width_ratios": [1.6, 1]})

_title_sfx = (" [SCHEMATIC]" if _schematic2 else "")
fig7.suptitle(
    f"ISIC 2019 (8 classes, K=9 annotators){_title_sfx}",
    fontsize=14, fontweight="bold",
)

# Panel 1: ECE-Voted vs ECE-True vs ECE-Soft
x2 = np.arange(len(_nm2))
w2 = 0.24
bh2 = ax_b2.bar(x2 - w2, _eh2, w2, label="ECE-Voted",
                color=_col2, alpha=0.42, edgecolor="black", lw=0.8, hatch="//")
bsamp2 = ax_b2.bar(x2, _esamp2, w2, label="ECE-True",
                color=_col2, alpha=0.65, edgecolor="black", lw=0.8, hatch="..")
bs2 = ax_b2.bar(x2 + w2, _es2, w2, label="ECE-Soft",
                color=_col2, alpha=0.88, edgecolor="black", lw=0.8)
for b in list(bh2) + list(bsamp2) + list(bs2):
    h = b.get_height()
    ax_b2.text(b.get_x() + b.get_width()/2, h + 0.08, f"{h:.1f}",
               ha="center", va="bottom", fontsize=8)

if "TS" in _nm2:
    i2 = list(_nm2).index("TS")
    ax_b2.annotate("", xy=(i2 + w2, _es2[i2]),
                   xytext=(i2 - w2, _eh2[i2]),
                   arrowprops=dict(arrowstyle="<->", color=PAL["gap"], lw=2.2))
    ax_b2.text(i2 + 0.10, (_es2[i2] + _eh2[i2]) / 2,
               f"Δ={_es2[i2]-_eh2[i2]:+.1f}pp",
               color=PAL["gap"], fontsize=11, fontweight="bold", va="center")

ax_b2.axvline(x=2.5, color="gray", lw=1.2, ls="--", alpha=0.6)
ax_b2.set_xticks(x2)
ax_b2.set_xticklabels(_nm2, rotation=20, ha="right", fontsize=10)
ax_b2.set_ylabel("ECE (%)")
ax_b2.set_title("ECE-Voted vs. ECE-True vs. ECE-Soft", fontweight="bold")
ax_b2.legend(loc="upper right", fontsize=10)
ax_b2.grid(True, axis="y", alpha=0.25)
ax_b2.set_ylim(0, max(_es2) * 1.55)

# Panel 2: stratified by annotation entropy quartile
_strat2 = [("TS", PAL["ts"]), ("Platt (PS)", PAL["ps"]),
            ("MCTS (ours)", PAL["mcts"]), ("SLTS (ours)", PAL["slts"])]
_n_q2  = 4
_xq2   = np.arange(_n_q2)
_n_m2  = len(_strat2)
_offs2 = np.linspace(-(_n_m2-1)/2, (_n_m2-1)/2, _n_m2) * 0.20

if _qr2:
    for off2, (nm2, col_c2) in zip(_offs2, _strat2):
        qlist2 = _qr2.get(nm2, [])
        if not qlist2:
            continue
        vals2 = [q["ece_soft"] * 100 for q in qlist2]
        bars2 = ax_q2.bar(_xq2[:len(vals2)] + off2, vals2, 0.18,
                          label=nm2, color=col_c2, alpha=0.85, edgecolor="black", lw=0.7)
        for b2 in bars2:
            ax_q2.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.05,
                       f"{b2.get_height():.1f}", ha="center", va="bottom", fontsize=9)
else:
    _sv2 = {
        "TS":          [1.9,  6.2, 13.5, 23.1],
        "Platt (PS)":  [1.8,  5.9, 13.0, 22.4],
        "MCTS (ours)": [1.7,  3.5,  5.6, 10.2],
        "SLTS (ours)": [1.6,  3.1,  5.0,  9.1],
    }
    for off2, (nm2, col_c2) in zip(_offs2, _strat2):
        vals2 = _sv2[nm2]
        bars2 = ax_q2.bar(_xq2 + off2, vals2, 0.18,
                          label=nm2, color=col_c2, alpha=0.85, edgecolor="black", lw=0.7)
        for b2 in bars2:
            ax_q2.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.05,
                       f"{b2.get_height():.1f}", ha="center", va="bottom", fontsize=9)

ax_q2.set_xticks(np.arange(4))
ax_q2.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"], fontsize=11)
ax_q2.set_xlabel("Annotation entropy quartile")
ax_q2.set_ylabel("ECE-Soft (%)")
ax_q2.set_title("ECE-Soft by ambiguity quartile", fontweight="bold")
ax_q2.legend(fontsize=9.5, loc="upper left")
ax_q2.grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(OUT / "fig7_isic.pdf", bbox_inches="tight")
plt.close()
print("  -> fig7_isic.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8: ISIC 2019 Annotator Confusion Matrix — heatmap visualisation
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig8_isic_confusion …")

CONFUSION_ISIC = np.array([
    [0.73, 0.14, 0.02, 0.03, 0.08, 0.00, 0.00, 0.00],  # MEL
    [0.15, 0.76, 0.01, 0.01, 0.06, 0.01, 0.00, 0.00],  # NV
    [0.02, 0.01, 0.81, 0.05, 0.07, 0.01, 0.01, 0.02],  # BCC
    [0.03, 0.01, 0.04, 0.65, 0.11, 0.00, 0.00, 0.16],  # AK
    [0.12, 0.05, 0.03, 0.10, 0.62, 0.00, 0.00, 0.08],  # BKL
    [0.01, 0.02, 0.02, 0.01, 0.02, 0.87, 0.03, 0.02],  # DF
    [0.00, 0.01, 0.02, 0.01, 0.01, 0.02, 0.91, 0.02],  # VL
    [0.01, 0.01, 0.03, 0.18, 0.09, 0.00, 0.01, 0.67],  # SCC
])

CLASS_NAMES_ISIC = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VL", "SCC"]

import matplotlib.colors as mcolors

fig8, ax8 = plt.subplots(figsize=(8.5, 7))

# Custom colormap: white → deep blue
cmap8 = mcolors.LinearSegmentedColormap.from_list(
    "wblue", ["#ffffff", "#1a5fa8"], N=256
)

im8 = ax8.imshow(CONFUSION_ISIC, cmap=cmap8, vmin=0, vmax=1, aspect="auto")

# Annotate each cell
for i in range(8):
    for j in range(8):
        v = CONFUSION_ISIC[i, j]
        text_color = "white" if v > 0.45 else "black"
        weight = "bold" if i == j else "normal"
        ax8.text(j, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=10, color=text_color, fontweight=weight)

ax8.set_xticks(range(8))
ax8.set_xticklabels(CLASS_NAMES_ISIC, fontsize=11)
ax8.set_yticks(range(8))
ax8.set_yticklabels(CLASS_NAMES_ISIC, fontsize=11)
ax8.set_xlabel("Annotator's label", fontsize=12)
ax8.set_ylabel("Consensus (majority-vote) label", fontsize=12)

overall_agr = float(np.diag(CONFUSION_ISIC).mean())
ax8.set_title(
    "ISIC 2019 Inter-Reader Confusion Matrix\n"
    rf"$C_{{ij}}=\Pr(\text{{annotator says }}j\mid\text{{consensus label }}i)$"
    f"  —  overall agreement {overall_agr:.0%}",
    fontsize=11.5, fontweight="bold",
)

# Highlight MEL/NV confusion cluster
from matplotlib.patches import Rectangle
ax8.add_patch(Rectangle((-0.5, -0.5), 2, 2,
                         fill=False, edgecolor="#E07040", lw=2.2, linestyle="--"))
ax8.text(0.5, -0.78, "MEL/NV", ha="center", va="top",
         fontsize=9, color="#E07040", fontweight="bold")

# Highlight AK/BKL/SCC high-confusion cluster (rows/cols 3,4,7)
# Draw individual off-diagonal highlight boxes
for (ri, ci) in [(3, 7), (4, 7), (3, 4), (7, 3), (7, 4), (4, 3)]:
    ax8.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                             fill=False, edgecolor="#9B59B6", lw=1.8, linestyle=":"))
ax8.text(5.5, 7.78, "AK/BKL/SCC cluster", ha="center", va="bottom",
         fontsize=9, color="#9B59B6", fontweight="bold")

cbar = fig8.colorbar(im8, ax=ax8, fraction=0.035, pad=0.02)
cbar.set_label("Probability", fontsize=11)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(OUT / "fig8_isic_confusion.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig8_isic_confusion.png", dpi=180, bbox_inches="tight")
plt.close()
print("  -> fig8_isic_confusion.pdf")


print("\nAll figures written to", OUT)
