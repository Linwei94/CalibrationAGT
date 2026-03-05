"""
Motivating Toy Example: Confidence Calibration under Ambiguous Ground Truth

Key question: Does Temperature Scaling (TS) fail under ambiguous ground truth?

Setup
-----
The realistic pipeline:
  1. Annotators disagree → we use the MAJORITY-VOTED (hard) label for training.
  2. A model is trained on those voted labels.
  3. TS is calibrated on a held-out set using the same voted labels.
  4. ECE is evaluated both with hard labels and with the true soft label distribution.

Key failure mode
----------------
For ambiguous inputs (where voted label is always "class 1", but the true
annotator distribution is π=[0, 0.55, 0.45]):
  - The model trained on voted labels has ~75% confidence for class 1.
  - TS sees: moderate confidence, ~100% hard accuracy → model looks UNDERconfident.
  - TS sets T < 1 → model becomes MORE confident (~90%+).
  - ECE-Soft: confidence ~90%, true soft accuracy 55% → large calibration gap.
  - SLTS targets the soft distribution directly: T >> 1 → confidence ≈ 55%.

This is the analog of the coverage gap in:
  Stutz et al. (2023). Conformal Prediction under Ambiguous Ground Truth. TMLR.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Use the shared calibration module if available
_EXP = Path(__file__).parent.parent / "experiments"
if _EXP.exists():
    sys.path.insert(0, str(_EXP))

np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(n_per_class=2000):
    """
    3-class dataset with strong label ambiguity in class 1.

    Class 0  (clear):     cluster at (-3, 0), π = [1,    0,    0   ]
    Class 1  (ambiguous): cluster at ( 0, 0), π = [0, 0.55, 0.45]
                          → voted label always = 1, but nearly half of
                            annotators label it as class 2.
    Class 2  (clear):     cluster at (+3, 0), π = [0,    0,    1   ]

    Key: clusters are well-separated (so the model is accurate), but the
    annotation distribution for class 1 is nearly 50/50 between classes 1 and 2.
    The model trained on voted labels achieves ~100% hard accuracy on class 1
    while having only moderate confidence (~75%).  TS interprets this moderate
    confidence + perfect accuracy as "underconfidence" and sets T < 1, pushing
    the model to ~90% confidence — but the true soft accuracy is only 55%,
    creating a large calibration gap.

    Training and calibration use the voted hard label.
    Test evaluation uses both hard and soft labels.
    """
    # Clear classes use tight isotropic clusters; ambiguous class is wider
    cov_clear = np.diag([0.6, 0.6])
    cov_amb   = np.diag([1.0, 0.6])   # wider x-spread → natural boundary uncertainty
    centers   = [np.array([-3., 0.]), np.array([0., 0.]), np.array([3., 0.])]
    PI_AMB    = [0., 0.55, 0.45]

    X_list, yhard_list, ysoft_list, amb_list = [], [], [], []
    for i, center in enumerate(centers):
        cov = cov_amb if i == 1 else cov_clear
        X_i = np.random.multivariate_normal(center, cov, n_per_class).astype(np.float32)
        if i == 0:
            yh = np.zeros(n_per_class, dtype=np.int64)
            ys = np.tile([1., 0., 0.], (n_per_class, 1)).astype(np.float32)
            am = np.zeros(n_per_class, dtype=bool)
        elif i == 1:
            yh = np.ones(n_per_class, dtype=np.int64)   # voted label always 1
            ys = np.tile(PI_AMB, (n_per_class, 1)).astype(np.float32)
            am = np.ones(n_per_class, dtype=bool)
        else:
            yh = np.full(n_per_class, 2, dtype=np.int64)
            ys = np.tile([0., 0., 1.], (n_per_class, 1)).astype(np.float32)
            am = np.zeros(n_per_class, dtype=bool)
        X_list.append(X_i); yhard_list.append(yh)
        ysoft_list.append(ys); amb_list.append(am)

    return (np.vstack(X_list),
            np.concatenate(yhard_list),
            np.vstack(ysoft_list).astype(np.float32),
            np.concatenate(amb_list))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )
    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    """Standard TS: NLL against hard (voted) labels."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits, labels_hard):
        opt  = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=300,
                                   tolerance_grad=1e-9, tolerance_change=1e-11)
        crit = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad(); loss = crit(self(logits), labels_hard); loss.backward(); return loss
        opt.step(closure)
        return self


class SoftLabelTS(nn.Module):
    """Soft-Label TS: KL(π(x) || softmax(z/T)) = cross-entropy with soft targets."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits, labels_soft):
        opt = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=300,
                                  tolerance_grad=1e-9, tolerance_change=1e-11)
        def closure():
            opt.zero_grad()
            loss = -(labels_soft * torch.log_softmax(self(logits), dim=1)).sum(1).mean()
            loss.backward(); return loss
        opt.step(closure)
        return self


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ece_bins(probs, targets, n_bins=10, min_count=5):
    """ECE with per-bin info. targets can be soft or 1-hot."""
    conf  = probs.max(1)
    pred  = probs.argmax(1)
    sacc  = targets[np.arange(len(pred)), pred]
    bins  = np.linspace(0., 1., n_bins + 1)
    ece   = 0.
    info  = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        n    = mask.sum()
        if n >= min_count:
            ac = float(conf[mask].mean()); aa = float(sacc[mask].mean())
            ece += (n / len(probs)) * abs(ac - aa)
            info.append((ac, aa, int(n)))
    return ece, info


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

C = dict(uncal='#5A8FC2', ts='#E07040', slts='#4CAF80',
         gap_over='#FFB3B3', gap_under='#C8F0C8')


def rel_diagram(ax, probs, targets, title, bar_color,
                n_bins=10, show_gap_ann=False):
    ece, info = ece_bins(probs, targets, n_bins)
    if not info:
        ax.set_title(f'{title}\nECE={ece:.3f}'); return ece
    confs, accs, _ = map(np.array, zip(*info))
    w = 0.9 / n_bins
    for c, a in zip(confs, accs):
        lo, hi = c - w/2, c + w/2
        col = C['gap_under'] if a > c else C['gap_over']
        ax.fill_between([lo, hi], [min(c,a), min(c,a)], [max(c,a), max(c,a)],
                        color=col, alpha=0.55, zorder=1)
    ax.bar(confs, accs, width=w*.82, color=bar_color, alpha=0.85, zorder=2)
    ax.plot([0,1],[0,1],'--', color='#333', lw=1.5, label='Perfect', zorder=3)
    if show_gap_ann:
        idx = int(np.argmax([abs(c-a) for c,a in zip(confs, accs)]))
        c0, a0 = confs[idx], accs[idx]
        ax.annotate('', xy=(c0, a0), xytext=(c0, c0),
                    arrowprops=dict(arrowstyle='<->', color='darkred', lw=2))
        ax.text(c0+.03, (c0+a0)/2, f'Gap\n{c0-a0:.2f}',
                color='darkred', fontsize=8, va='center')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Confidence', fontsize=10)
    ax.set_ylabel('Avg. label prob.', fontsize=10)
    ax.set_title(f'{title}\nECE = {ece:.3f}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)
    return ece


def ann_T(ax, T, label):
    ax.text(.97,.04, f'T={T:.3f}\n({label})', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.85))


def val_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+.003, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Data
    X, y_hard, y_soft, ambiguous = generate_data(n_per_class=1200)
    idx = np.arange(len(X))
    idx_tr, idx_rest = train_test_split(idx, test_size=.4, random_state=0)
    idx_cal, idx_te  = train_test_split(idx_rest, test_size=.5, random_state=0)

    X_tr, yh_tr           = X[idx_tr],  y_hard[idx_tr]
    X_cal, yh_cal, ys_cal = X[idx_cal], y_hard[idx_cal], y_soft[idx_cal]
    X_te,  yh_te,  ys_te  = X[idx_te],  y_hard[idx_te],  y_soft[idx_te]
    amb_te = ambiguous[idx_te]

    Xtr_t = torch.FloatTensor(X_tr); ytr_t = torch.LongTensor(yh_tr)
    Xcl_t = torch.FloatTensor(X_cal); ycl_t = torch.LongTensor(yh_cal)
    yscl_t = torch.FloatTensor(ys_cal); Xte_t = torch.FloatTensor(X_te)

    # Train on voted labels — use moderate regularisation to keep confidence
    # in the 70–85% range (not near-saturated), so TS can shift it meaningfully.
    model = MLP()
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
    crit  = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=256, shuffle=True)
    for _ in range(150):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        lg_cal = model(Xcl_t); lg_te = model(Xte_t)

    # Calibrate
    ts   = TemperatureScaling(); ts.fit(lg_cal,  ycl_t)
    slts = SoftLabelTS();        slts.fit(lg_cal, yscl_t)
    T_ts, T_slts = ts.temperature.item(), slts.temperature.item()

    # Platt scaling and hard histogram binning via shared module (if available)
    try:
        from calibration import PlattScaling, HardHistogramBinning, apply_parametric
        ps   = PlattScaling(3).fit(lg_cal, ycl_t)
        hb   = HardHistogramBinning(n_bins=10).fit(
                   torch.softmax(lg_cal, 1).numpy(), yh_cal)
        _have_extra = True
    except ImportError:
        _have_extra = False

    with torch.no_grad():
        p_raw  = torch.softmax(lg_te,        1).numpy()
        p_ts   = torch.softmax(ts(lg_te),    1).numpy()
        p_slts = torch.softmax(slts(lg_te),  1).numpy()

    if _have_extra:
        p_ps = apply_parametric(ps, lg_te.numpy())
        cal_conf, pred = hb.calibrate(p_raw)
        p_hb = np.zeros_like(p_raw)
        for i in range(len(p_raw)):
            p_hb[i, pred[i]] = cal_conf[i]
            rest = (1 - cal_conf[i]) / 2
            for j in range(3):
                if j != pred[i]:
                    p_hb[i, j] = rest

    yh1hot = np.eye(3)[yh_te]

    # Print summary
    print("=" * 62)
    print(f"  Temperature (TS):          T = {T_ts:.4f}  {'< 1 → MORE confident!' if T_ts < 1 else '> 1 → less confident'}")
    print(f"  Temperature (SLTS, ours):  T = {T_slts:.4f}")
    print("=" * 62)
    print(f"  {'Method':<18} | ECE-Hard | ECE-Soft |  Gap Δ")
    print("  " + "-"*55)
    all_probs = [("Uncalibrated  ", p_raw), ("TS            ", p_ts), ("SLTS (ours)   ", p_slts)]
    if _have_extra:
        all_probs.insert(2, ("Platt (PS)    ", p_ps))
        all_probs.insert(3, ("HB-Hard       ", p_hb))
    for name, p in all_probs:
        eh, _ = ece_bins(p, yh1hot); es, _ = ece_bins(p, ys_te)
        print(f"  {name} | {eh:.4f}   | {es:.4f}   | {es-eh:+.4f}")
    print("\n  --- Stratified ECE-Soft ---")
    for name, p in [("TS          ", p_ts), ("SLTS (ours) ", p_slts)]:
        ea, _ = ece_bins(p[amb_te],  ys_te[amb_te])
        ec, _ = ece_bins(p[~amb_te], ys_te[~amb_te])
        print(f"  {name}: ambiguous = {ea:.4f}   clear = {ec:.4f}")

    # ------------------------------------------------------------------ Figure
    _make_figure(
        X_te, yh_te, ys_te, amb_te, yh1hot,
        p_raw, p_ts, p_slts,
        p_ps if _have_extra else None,
        p_hb if _have_extra else None,
        T_ts, T_slts,
    )


def _make_figure(X_te, yh_te, ys_te, amb_te, yh1hot,
                 p_raw, p_ts, p_slts, p_ps, p_hb, T_ts, T_slts):
    """
    Paper-quality figure.  Layout (2 rows, 4 cols):
      Row 1: data scatter | soft reliability (Uncal) | soft reliability (TS) | soft reliability (SLTS)
      Row 2: (span 3) ECE-Hard vs ECE-Soft bar chart | stratified ECE bar chart
    """
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # ── colours ──────────────────────────────────────────────────────────────
    COL = {
        'uncal': '#888888',
        'ts':    '#D45B3A',
        'platt': '#E8904A',
        'hb':    '#C04030',
        'slts':  '#3A9E6A',
        'over':  '#FADADD',
        'under': '#D4EDDA',
        'diag':  '#333333',
        'c0':    '#5A8FC2',
        'c1amb': '#E0A030',
        'c2':    '#9B59B6',
    }

    fig = plt.figure(figsize=(13.5, 7.5))
    gs  = fig.add_gridspec(
        2, 4,
        hspace=0.52, wspace=0.40,
        height_ratios=[1.0, 0.9],
        left=0.06, right=0.98, top=0.91, bottom=0.10,
    )
    ax_data  = fig.add_subplot(gs[0, 0])
    ax_r0    = fig.add_subplot(gs[0, 1])
    ax_r1    = fig.add_subplot(gs[0, 2])
    ax_r2    = fig.add_subplot(gs[0, 3])
    ax_bar   = fig.add_subplot(gs[1, 0:3])
    ax_strat = fig.add_subplot(gs[1, 3])

    fig.suptitle(
        r'Calibration under Ambiguous Ground Truth — Motivating Toy Example'
        '\n'
        r'Class 1 annotator distribution $\pi=[0,\,0.55,\,0.45]$: '
        r'55\% vote class 1, 45\% vote class 2.  '
        r'Model and TS calibrated on VOTED labels.',
        fontsize=9.5, fontweight='bold',
    )

    # ── (a) Data scatter ─────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    sample = rng.choice(len(X_te), size=min(600, len(X_te)), replace=False)
    Xs, ys_hard = X_te[sample], yh_te[sample]
    for cls, col, lbl in [
            (0, COL['c0'],   r'Class 0 — $\pi=[1,0,0]$'),
            (1, COL['c1amb'], r'Class 1 — $\pi=[0,0.55,0.45]$'),
            (2, COL['c2'],   r'Class 2 — $\pi=[0,0,1]$')]:
        mask = ys_hard == cls
        ax_data.scatter(Xs[mask, 0], Xs[mask, 1],
                        c=col, s=12, alpha=0.55, edgecolors='none', label=lbl)

    # Draw π pie icons at cluster centres
    pie_centers = [(-3, 1.8), (0, 1.8), (3, 1.8)]
    pie_pis     = [[1, 0, 0], [0, 0.55, 0.45], [0, 0, 1]]
    pie_colors  = [[COL['c0']], [COL['c1amb'], COL['c2']], [COL['c2']]]
    for (cx, cy), pi, cols in zip(pie_centers, pie_pis, pie_colors):
        non_zero = [(p, c) for p, c in zip(pi, [COL['c0'], COL['c1amb'], COL['c2']]) if p > 0]
        pvals = [x[0] for x in non_zero]; pcols = [x[1] for x in non_zero]
        ax_data.pie(pvals, colors=pcols, radius=0.55, center=(cx, cy),
                    wedgeprops=dict(linewidth=0.5, edgecolor='white'))

    ax_data.set_xlim(-5.5, 5.5); ax_data.set_ylim(-2.8, 2.8)
    ax_data.set_xlabel('Feature $x_1$'); ax_data.set_ylabel('Feature $x_2$')
    ax_data.set_title('(a) Data & annotator distribution $\\pi$\n(pie = label distribution per cluster)',
                      fontweight='bold')
    ax_data.legend(loc='lower center', framealpha=0.8, ncol=1,
                   bbox_to_anchor=(0.5, -0.32))

    # ── helper: reliability diagram ──────────────────────────────────────────
    def rel_diag(ax, probs, targets, title, bar_col, n_bins=10, note=None):
        ece, info = ece_bins(probs, targets, n_bins)
        if info:
            confs, accs, _ = map(np.array, zip(*info))
            w = 0.80 / n_bins
            for c, a in zip(confs, accs):
                color = COL['over'] if c > a else COL['under']
                ax.fill_between([c - w/2, c + w/2],
                                [min(c, a)]*2, [max(c, a)]*2,
                                color=color, zorder=1)
            ax.bar(confs, accs, width=w * 0.78, color=bar_col,
                   alpha=0.88, zorder=2, linewidth=0)
        ax.plot([0, 1], [0, 1], '--', color=COL['diag'], lw=1.2, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence'); ax.set_ylabel('Avg. annotator prob.')
        ax.set_title(f'{title}\nECE-Soft = {ece*100:.1f}%', fontweight='bold')
        ax.grid(True, alpha=0.20, linewidth=0.5)
        if note:
            ax.text(0.97, 0.04, note, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.3', fc='#FFFBE6',
                              ec='#BBAA33', alpha=0.92))
        return ece

    # ── (b)-(d) Soft-label reliability diagrams ───────────────────────────────
    rel_diag(ax_r0, p_raw, ys_te,
             '(b) Uncalibrated', COL['uncal'])
    rel_diag(ax_r1, p_ts,  ys_te,
             f'(c) TS  [hard-label cal.]', COL['ts'],
             note=f'$T={T_ts:.2f}<1$\n↑ confidence ↑ gap')
    rel_diag(ax_r2, p_slts, ys_te,
             f'(d) SLTS [soft-label cal.]', COL['slts'],
             note=f'$T={T_slts:.2f}>1$\n↓ confidence ↓ gap')

    # Annotate TS gap
    ece_ts, info_ts = ece_bins(p_ts, ys_te, 10)
    if info_ts:
        confs_ts, accs_ts, _ = map(np.array, zip(*info_ts))
        worst_i = np.argmax(np.abs(confs_ts - accs_ts))
        c0, a0  = confs_ts[worst_i], accs_ts[worst_i]
        ax_r1.annotate('', xy=(c0, a0), xytext=(c0, c0),
                       arrowprops=dict(arrowstyle='<->', color='#AA1010', lw=1.8))
        ax_r1.text(c0 + 0.04, (c0 + a0) / 2,
                   f'Δ={abs(c0-a0):.2f}',
                   color='#AA1010', fontsize=8, va='center', fontweight='bold')

    # Add legend for shading to (d)
    ax_r2.legend(
        handles=[mpatches.Patch(color=COL['over'],  alpha=0.8, label='Overconfident'),
                 mpatches.Patch(color=COL['under'], alpha=0.8, label='Underconfident')],
        loc='upper left', framealpha=0.85)

    # ── (e) ECE bar chart ────────────────────────────────────────────────────
    methods_bar = [('Uncalibrated', p_raw,  COL['uncal']),
                   ('TS',           p_ts,   COL['ts']),
                   ('Platt (PS)',   p_ps,   COL['platt']),
                   ('HB-Hard',      p_hb,   COL['hb']),
                   ('SLTS (ours)',  p_slts, COL['slts'])]
    if p_ps is None:
        methods_bar = [(n, p, c) for n, p, c in methods_bar
                       if n not in ('Platt (PS)', 'HB-Hard')]

    bar_names, eh_vals, es_vals = [], [], []
    for nm, p, _ in methods_bar:
        eh, _ = ece_bins(p, yh1hot, 10)
        es, _ = ece_bins(p, ys_te,  10)
        bar_names.append(nm); eh_vals.append(eh * 100); es_vals.append(es * 100)

    x  = np.arange(len(bar_names)); w = 0.32
    bar_cols = [c for _, _, c in methods_bar]

    bh = ax_bar.bar(x - w/2, eh_vals, w, color=bar_cols, alpha=0.45,
                    edgecolor='black', lw=0.7, hatch='//', label='ECE-Hard $\\downarrow$ (vs. voted labels)')
    bs = ax_bar.bar(x + w/2, es_vals, w, color=bar_cols, alpha=0.92,
                    edgecolor='black', lw=0.7,        label='ECE-Soft $\\downarrow$ (vs. annotator dist.)')

    # Value labels on bars
    for b in list(bh) + list(bs):
        h = b.get_height()
        ax_bar.text(b.get_x() + b.get_width()/2, h + 0.25,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=7.5)

    # Δ arrow for TS
    ts_i = bar_names.index('TS')
    gap_ts = es_vals[ts_i] - eh_vals[ts_i]
    ax_bar.annotate('', xy=(ts_i + w/2, es_vals[ts_i]),
                    xytext=(ts_i - w/2, eh_vals[ts_i]),
                    arrowprops=dict(arrowstyle='<->', color='#AA1010', lw=2.0))
    ax_bar.text(ts_i + 0.08, (es_vals[ts_i] + eh_vals[ts_i])/2,
                f'$\\Delta={gap_ts:+.1f}$pp',
                color='#AA1010', fontsize=9, va='center', fontweight='bold')

    # Δ label for Uncal
    unc_i = bar_names.index('Uncalibrated')
    gap_unc = es_vals[unc_i] - eh_vals[unc_i]
    ax_bar.text(unc_i, max(es_vals[unc_i], eh_vals[unc_i]) + 1.5,
                f'$\\Delta={gap_unc:+.1f}$pp',
                ha='center', color='#555555', fontsize=8)

    ax_bar.set_xticks(x); ax_bar.set_xticklabels(bar_names)
    ax_bar.set_ylabel('ECE (%)'); ax_bar.set_ylim(0, max(es_vals) * 1.55)
    ax_bar.set_title(
        '(e) Hard- vs. soft-label ECE — calibration gap $\\Delta = $ ECE-Soft $-$ ECE-Hard\n'
        'Hard-label methods reduce ECE-Hard but widen $\\Delta$;  SLTS closes it',
        fontweight='bold')
    ax_bar.legend(loc='upper right', framealpha=0.88)
    ax_bar.grid(True, alpha=0.25, axis='y', linewidth=0.5)

    # ── (f) Stratified bar chart ──────────────────────────────────────────────
    ea_ts,   _ = ece_bins(p_ts[amb_te],   ys_te[amb_te],   10)
    ec_ts,   _ = ece_bins(p_ts[~amb_te],  ys_te[~amb_te],  10)
    ea_sl,   _ = ece_bins(p_slts[amb_te], ys_te[amb_te],   10)
    ec_sl,   _ = ece_bins(p_slts[~amb_te], ys_te[~amb_te], 10)

    x2 = np.arange(2); w2 = 0.30
    for ii, (nm, ea, ec, col) in enumerate([
            ('TS',          ea_ts*100, ec_ts*100, COL['ts']),
            ('SLTS (ours)', ea_sl*100, ec_sl*100, COL['slts'])]):
        off = (ii - 0.5) * (w2 + 0.05)
        b   = ax_strat.bar(x2 + off, [ea, ec], w2, label=nm,
                           color=col, alpha=0.88, edgecolor='black', lw=0.7)
        for bar in b:
            h = bar.get_height()
            ax_strat.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                          f'{h:.1f}', ha='center', va='bottom', fontsize=7.5)

    ax_strat.set_xticks(x2)
    ax_strat.set_xticklabels(['Ambiguous\nexamples', 'Clear\nexamples'])
    ax_strat.set_ylabel('ECE-Soft (%)')
    ax_strat.set_title('(f) ECE-Soft\nstratified by ambiguity', fontweight='bold')
    ax_strat.legend(loc='upper right'); ax_strat.grid(True, alpha=0.25, axis='y', linewidth=0.5)
    ax_strat.set_ylim(0, max(ea_ts, ea_sl) * 100 * 1.45)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_png = Path(__file__).parent / 'toy_example_figure.png'
    out_pdf = Path(__file__).parent.parent / 'paper' / 'figs' / 'fig1_toy.pdf'
    out_png2 = Path(__file__).parent.parent / 'paper' / 'figs' / 'fig1_toy.png'

    plt.savefig(str(out_png), dpi=150, bbox_inches='tight')
    plt.savefig(str(out_pdf), bbox_inches='tight')
    plt.savefig(str(out_png2), dpi=150, bbox_inches='tight')
    print(f'\nFigure saved →\n  {out_png}\n  {out_pdf}')
    plt.close()


if __name__ == '__main__':
    main()
