"""
Motivating toy example: calibration under ambiguous ground truth.

The pipeline mirrors standard post-hoc calibration:
  1. Annotators disagree on one cluster, but the majority-voted label is fixed.
  2. The classifier is trained on those voted labels.
  3. Standard calibrators are fitted on a held-out calibration split using the
     same voted labels.
  4. We compare voted-label ECE and true-label ECE under the underlying
     ambiguity-aware label distribution.
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

# Use the shared calibration module from the experiments package.
_EXP = Path(__file__).parent
if _EXP.exists():
    sys.path.insert(0, str(_EXP))

np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(n_per_class=2000):
    """
    Three Gaussian clusters with one genuinely ambiguous class.

    x | y
    -------
    x ~ N(mu_0, Sigma_0), mu_0 = (-3.2,  1.1),  pi = [1.0, 0.0, 0.0]
    x ~ N(mu_1, Sigma_1), mu_1 = ( 0.0,  0.0),  pi = [0.0, 0.7, 0.3]
    x ~ N(mu_2, Sigma_2), mu_2 = ( 3.2, -1.1),  pi = [0.0, 0.0, 1.0]

    The voted label for the ambiguous cluster is always class 1 because
    pi_1 > pi_2.  Standard calibrators therefore interpret these examples as
    nearly perfectly correct but slightly underconfident, which pushes them
    toward lower temperatures and higher confidence.  Under the true label
    distribution, that extra confidence is miscalibrated.
    """
    cov_clear = np.diag([0.60, 0.45])
    cov_amb = np.diag([1.15, 0.75])
    centers = [
        np.array([-3.2, 1.1]),
        np.array([0.0, 0.0]),
        np.array([3.2, -1.1]),
    ]
    PI_AMB = [0.0, 0.70, 0.30]

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
    """Standard TS: NLL against hard (voted) labels. T = exp(log_T) > 0."""
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels_hard):
        opt  = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=300,
                                   tolerance_grad=1e-9, tolerance_change=1e-11)
        crit = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad(); loss = crit(self(logits), labels_hard); loss.backward(); return loss
        opt.step(closure)
        return self


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ece_bins(probs, targets, n_bins=10, min_count=5):
    """ECE with per-bin info for one-hot or distributional targets."""
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


def sampled_true_ece(probs, target_dist, n_bins=10, n_trials=100, seed=0):
    """Monte Carlo estimate of ECE_true under repeated draws from pi(.|x)."""
    rng = np.random.default_rng(seed)
    n, k = target_dist.shape
    total = 0.0
    for _ in range(n_trials):
        sampled = np.array([rng.choice(k, p=target_dist[i]) for i in range(n)])
        sampled_1hot = np.eye(k)[sampled]
        e, _ = ece_bins(probs, sampled_1hot, n_bins=n_bins)
        total += e
    return total / n_trials


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Data
    X, y_hard, y_soft, ambiguous = generate_data(n_per_class=800)
    idx = np.arange(len(X))
    idx_tr, idx_rest = train_test_split(idx, test_size=.4, random_state=0)
    idx_cal, idx_te  = train_test_split(idx_rest, test_size=.5, random_state=0)

    X_tr, yh_tr           = X[idx_tr],  y_hard[idx_tr]
    X_cal, yh_cal = X[idx_cal], y_hard[idx_cal]
    X_te,  yh_te,  ys_te  = X[idx_te],  y_hard[idx_te],  y_soft[idx_te]
    amb_te = ambiguous[idx_te]

    Xtr_t = torch.FloatTensor(X_tr); ytr_t = torch.LongTensor(yh_tr)
    Xcl_t = torch.FloatTensor(X_cal); ycl_t = torch.LongTensor(yh_cal)
    Xte_t = torch.FloatTensor(X_te)

    # Partially train: model becomes somewhat overconfident on voted labels
    # (ECE_voted ~3%), giving TS room to reduce confidence to ~1%.
    model = MLP()
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=2e-2)
    crit  = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=128, shuffle=True)
    for _ in range(150):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        lg_cal = model(Xcl_t); lg_te = model(Xte_t)

    # Calibrate
    ts   = TemperatureScaling(); ts.fit(lg_cal,  ycl_t)
    T_ts = ts.temperature.item()

    # Platt scaling and hard histogram binning via shared module (if available)
    try:
        from calibration import PlattScaling, HardHistogramBinning, apply_parametric
        ps   = PlattScaling(3).fit(lg_cal, ycl_t)
        hb   = HardHistogramBinning(n_bins=6).fit(
                   torch.softmax(lg_cal, 1).numpy(), yh_cal)
        _have_extra = True
    except ImportError:
        _have_extra = False

    with torch.no_grad():
        p_raw  = torch.softmax(lg_te,        1).numpy()
        p_ts   = torch.softmax(ts(lg_te),    1).numpy()
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
    print(f"  Temperature (TS):      T = {T_ts:.4f}  {'< 1 -> MORE confident' if T_ts < 1 else '> 1 -> less confident'}")
    print("=" * 62)
    print(f"  {'Method':<18} | ECE_voted | ECE_true")
    print("  " + "-" * 60)
    all_probs = [("Uncalibrated  ", p_raw), ("TS            ", p_ts)]
    if _have_extra:
        all_probs.insert(2, ("Platt (PS)    ", p_ps))
        all_probs.insert(3, ("HB-Hard       ", p_hb))
    for name, p in all_probs:
        eh, _ = ece_bins(p, yh1hot)
        et = sampled_true_ece(p, ys_te)
        print(f"  {name} | {eh:.4f}    | {et:.4f}")
    print("\n  --- Stratified ECE_true ---")
    strat_probs = [("TS          ", p_ts)]
    if _have_extra:
        strat_probs.extend([("Platt (PS)  ", p_ps), ("HB-Hard     ", p_hb)])
    for name, p in strat_probs:
        ea = sampled_true_ece(p[amb_te], ys_te[amb_te])
        ec = sampled_true_ece(p[~amb_te], ys_te[~amb_te])
        print(f"  {name}: ambiguous = {ea:.4f}   clear = {ec:.4f}")

    # ------------------------------------------------------------------ Figure
    _make_figure(
        X_te, yh_te, ys_te, amb_te, yh1hot,
        p_raw, p_ts,
        p_ps if _have_extra else None,
        p_hb if _have_extra else None,
    )


def _make_figure(X_te, yh_te, ys_te, amb_te, yh1hot,
                 p_raw, p_ts, p_ps, p_hb):
    """Paper-quality figure for voted-label vs true-label calibration."""
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # ── colours ──────────────────────────────────────────────────────────────
    COL = {
        'uncal': '#888888',
        'ts':    '#D45B3A',
        'platt': '#4C78A8',
        'hb':    '#2A9D8F',
        'over':  '#FADADD',
        'under': '#D4EDDA',
        'diag':  '#333333',
        'c0':    '#5A8FC2',
        'c1amb': '#E0A030',
        'c1min': '#E74C3C',   # 30% minority label in middle cluster
        'c2':    '#9B59B6',
    }

    fig = plt.figure(figsize=(14.6, 5.2))
    gs  = fig.add_gridspec(
        1, 3,
        width_ratios=[1.00, 1.5, 1.00],
        wspace=0.08,
        left=0.045, right=0.99, top=0.82, bottom=0.17,
    )
    ax_data  = fig.add_subplot(gs[0, 0])
    ax_summary = fig.add_subplot(gs[0, 1])
    ax_strat = fig.add_subplot(gs[0, 2])

    # ── (a) Data scatter ─────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    sample = rng.choice(len(X_te), size=min(420, len(X_te)), replace=False)
    Xs, ys_hard = X_te[sample], yh_te[sample]
    mask0 = ys_hard == 0
    mask1 = ys_hard == 1
    mask2 = ys_hard == 2
    amb_idx = np.where(mask1)[0]
    amb_perm = rng.permutation(amb_idx)
    split = int(round(0.70 * len(amb_perm)))
    amb_major = amb_perm[:split]
    amb_minor = amb_perm[split:]

    ax_data.scatter(Xs[mask0, 0], Xs[mask0, 1],
                    c=COL['c0'], s=10, alpha=0.55, edgecolors='none',
                    label=r'Class 0: $\pi=[1,0,0]$')
    ax_data.scatter(Xs[amb_major, 0], Xs[amb_major, 1],
                    c=COL['c1amb'], s=12, alpha=0.60, edgecolors='none',
                    label='Middle 70%: label 1')
    ax_data.scatter(Xs[amb_minor, 0], Xs[amb_minor, 1],
                    c=COL['c1min'], s=12, alpha=0.75, edgecolors='none',
                    label='Middle 30%: label 2')
    ax_data.scatter(Xs[mask2, 0], Xs[mask2, 1],
                    c=COL['c2'], s=10, alpha=0.35, edgecolors='none',
                    label=r'Class 2: $\pi=[0,0,1]$')

    # Draw π pie icons at cluster centres
    pie_centers = [(-3.2, 2.2), (0.0, 1.9), (3.2, 0.2)]
    pie_pis = [[1, 0, 0], [0, 0.70, 0.30], [0, 0, 1]]
    pie_colors = [[COL['c0']], [COL['c1amb'], COL['c1min']], [COL['c2']]]
    for (cx, cy), pi, cols in zip(pie_centers, pie_pis, pie_colors):
        non_zero = [(p, c) for p, c in zip(pi, [COL['c0'], COL['c1amb'], COL['c2']]) if p > 0]
        pvals = [x[0] for x in non_zero]; pcols = [x[1] for x in non_zero]
        ax_data.pie(pvals, colors=pcols, radius=0.44, center=(cx, cy),
                    wedgeprops=dict(linewidth=0.5, edgecolor='white'))

    ax_data.set_xlim(-5.8, 5.8)
    ax_data.set_ylim(-3.2, 3.0)
    ax_data.set_box_aspect(1)
    ax_data.set_xlabel('Feature $x_1$'); ax_data.set_ylabel('Feature $x_2$')
    ax_data.set_title(
        '(a) Data',
        fontweight='bold', y=1.03, pad=1
    )
    ax_data.legend(loc='lower left', framealpha=0.88, ncol=1,
                   bbox_to_anchor=(0.02, 0.02), borderaxespad=0.0,
                   handlelength=1.2, borderpad=0.35, labelspacing=0.35)

    # ── (b) Summary chart replacing Table 1 ─────────────────────────────────
    summary_entries = [('Uncal.', p_raw, COL['uncal']), ('TS', p_ts, COL['ts'])]
    if p_ps is not None:
        summary_entries.append(('Platt', p_ps, COL['platt']))
    if p_hb is not None:
        summary_entries.append(('HB', p_hb, COL['hb']))

    x1 = np.arange(len(summary_entries))
    w1 = 0.46
    voted_vals, true_vals = [], []
    for _, probs, _ in summary_entries:
        e_voted, _ = ece_bins(probs, yh1hot)
        e_true = sampled_true_ece(probs, ys_te)
        voted_vals.append(e_voted * 100)
        true_vals.append(e_true * 100)

    voted_bars = ax_summary.bar(
        x1 - w1 / 2, voted_vals, width=w1,
        color='white', edgecolor=[c for _, _, c in summary_entries],
        linewidth=1.2, hatch='///', label=r'$\mathrm{ECE}_{\mathrm{voted}}$'
    )
    true_bars = ax_summary.bar(
        x1 + w1 / 2, true_vals, width=w1,
        color=[c for _, _, c in summary_entries], alpha=0.88,
        edgecolor='black', linewidth=0.7, label=r'$\mathrm{ECE}_{\mathrm{true}}$'
    )
    for bars in (voted_bars, true_bars):
        for bar in bars:
            h = bar.get_height()
            ax_summary.text(
                bar.get_x() + bar.get_width() / 2, h + 0.22, f'{h:.2f}',
                ha='center', va='bottom', fontsize=11.5
            )

    ax_summary.set_xticks(x1)
    ax_summary.set_xticklabels([name for name, _, _ in summary_entries])
    ax_summary.set_ylabel('ECE (%)')
    ax_summary.set_title('(e) ECE summary', fontweight='bold', pad=8)
    ax_summary.legend(loc='upper left', framealpha=0.88, ncol=2)
    ax_summary.grid(True, alpha=0.25, axis='y', linewidth=0.5)
    ax_summary.set_ylim(0, max(true_vals) * 1.22)
    ax_summary.set_box_aspect(0.95)

    # ── (d) Stratified bar chart for standard calibrators ───────────────────
    strat_entries = [('TS', p_ts, COL['ts'])]
    if p_ps is not None:
        strat_entries.append(('Platt', p_ps, COL['platt']))
    if p_hb is not None:
        strat_entries.append(('HistBin', p_hb, COL['hb']))

    x2 = np.arange(2)
    w2 = 0.22
    offsets = np.linspace(-w2, w2, len(strat_entries))
    ymax = 0.0
    for off, (nm, probs, col) in zip(offsets, strat_entries):
        ea = sampled_true_ece(probs[amb_te], ys_te[amb_te], n_bins=10) * 100
        ec = sampled_true_ece(probs[~amb_te], ys_te[~amb_te], n_bins=10) * 100
        ymax = max(ymax, ea, ec)
        bars = ax_strat.bar(x2 + off, [ea, ec], w2, label=nm,
                            color=col, alpha=0.88, edgecolor='black', lw=0.7)
        for bar in bars:
            h = bar.get_height()
            ax_strat.text(bar.get_x() + bar.get_width()/2, h + 0.45,
                          f'{h:.1f}', ha='center', va='bottom', fontsize=11.5)

    ax_strat.set_xticks(x2)
    ax_strat.set_xticklabels(['Ambiguous\nexamples', 'Clear\nexamples'])
    ax_strat.set_ylabel('ECE_true (%)')
    ax_strat.set_title('(f) Stratified ECE', fontweight='bold', pad=8)
    ax_strat.legend(loc='upper right', framealpha=0.88)
    ax_strat.grid(True, alpha=0.25, axis='y', linewidth=0.5)
    ax_strat.set_ylim(0, ymax * 1.35)
    ax_strat.set_box_aspect(1)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_png = Path(__file__).parent / 'toy_example_figure.png'
    out_pdf = Path(__file__).parent.parent / 'paper' / 'figs' / 'toy_example.pdf'
    out_png2 = Path(__file__).parent.parent / 'paper' / 'figs' / 'toy_example.png'

    plt.savefig(str(out_png), dpi=150, bbox_inches='tight')
    plt.savefig(str(out_pdf), bbox_inches='tight')
    plt.savefig(str(out_png2), dpi=150, bbox_inches='tight')
    print(f'\nFigure saved →\n  {out_png}\n  {out_pdf}')
    plt.close()


if __name__ == '__main__':
    main()
