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
annotator distribution is π=[0, 0.7, 0.3]):
  - The model trained on voted labels becomes confident about class 1.
  - TS sees: high confidence, 100% hard accuracy → model looks UNDERconfident.
  - TS decreases T (< 1) → model becomes MORE confident.
  - ECE-Soft: confidence ~90%, true soft accuracy 70% → calibration gap.

This is the analog of the coverage gap in:
  Stutz et al. (2023). Conformal Prediction under Ambiguous Ground Truth. TMLR.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(n_per_class=1200):
    """
    3-class dataset with label ambiguity in class 1.

    Class 0  (clear):     cluster at (-3, 0), π = [1,   0,   0 ]
    Class 1  (ambiguous): cluster at ( 0, 0), π = [0, 0.7, 0.3]
                          → voted label always = 1
    Class 2  (clear):     cluster at (+3, 0), π = [0,   0,   1 ]

    Training and calibration use the voted hard label.
    Test evaluation uses both hard and soft labels.
    """
    cov     = np.diag([1.0, 0.5])
    centers = [np.array([-3., 0.]), np.array([0., 0.]), np.array([3., 0.])]
    PI_AMB  = [0., 0.7, 0.3]

    X_list, yhard_list, ysoft_list, amb_list = [], [], [], []
    for i, center in enumerate(centers):
        X_i = np.random.multivariate_normal(center, cov, n_per_class).astype(np.float32)
        if i == 0:
            yh = np.zeros(n_per_class, dtype=np.int64)
            ys = np.tile([1., 0., 0.], (n_per_class, 1)).astype(np.float32)
            am = np.zeros(n_per_class, dtype=bool)
        elif i == 1:
            yh = np.ones(n_per_class, dtype=np.int64)            # voted label
            ys = np.tile(PI_AMB,       (n_per_class, 1)).astype(np.float32)
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
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3),
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

    # Train on voted labels
    model = MLP()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=128, shuffle=True)
    for _ in range(200):
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

    with torch.no_grad():
        p_raw  = torch.softmax(lg_te,        1).numpy()
        p_ts   = torch.softmax(ts(lg_te),    1).numpy()
        p_slts = torch.softmax(slts(lg_te),  1).numpy()

    yh1hot = np.eye(3)[yh_te]

    # Print summary
    print("=" * 62)
    print(f"  Temperature (TS):          T = {T_ts:.4f}  {'< 1 → MORE confident!' if T_ts < 1 else '> 1 → less confident'}")
    print(f"  Temperature (SLTS, ours):  T = {T_slts:.4f}")
    print("=" * 62)
    print(f"  {'Method':<18} | ECE-Hard | ECE-Soft")
    print("  " + "-"*44)
    all_probs = [("Uncalibrated  ", p_raw), ("TS            ", p_ts), ("SLTS (ours)   ", p_slts)]
    for name, p in all_probs:
        eh, _ = ece_bins(p, yh1hot); es, _ = ece_bins(p, ys_te)
        print(f"  {name} | {eh:.4f}   | {es:.4f}")
    print("\n  --- Stratified ECE-Soft ---")
    for name, p in [("TS          ", p_ts), ("SLTS (ours) ", p_slts)]:
        ea, _ = ece_bins(p[amb_te],  ys_te[amb_te])
        ec, _ = ece_bins(p[~amb_te], ys_te[~amb_te])
        print(f"  {name}: ambiguous = {ea:.4f}   clear = {ec:.4f}")

    # ------------------------------------------------------------------ Figure
    fig = plt.figure(figsize=(15, 13))
    fig.suptitle(
        'Calibration under Ambiguous Ground Truth — Motivating Toy Example\n'
        'Model trained on VOTED labels.  TS calibrated on VOTED labels.',
        fontsize=13, fontweight='bold', y=1.00)

    gs   = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.35,
                             height_ratios=[1, 1, 0.85])
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    ax_ov   = fig.add_subplot(gs[2, 0:2])    # overall ECE bar chart
    ax_strat = fig.add_subplot(gs[2, 2])     # stratified ECE bar chart

    methods = [("Uncalibrated", p_raw,  C['uncal'], None),
               ("TS",           p_ts,   C['ts'],    T_ts),
               ("SLTS (ours)", p_slts,  C['slts'],  T_slts)]

    eh_vals, es_vals, names, ea_vals, ec_vals = [], [], [], [], []

    for col, (name, p, color, T) in enumerate(methods):
        eh = rel_diagram(axes[0][col], p, yh1hot,
                          f'{name}  [Hard labels]',  color, n_bins=10)
        es = rel_diagram(axes[1][col], p, ys_te,
                          f'{name}  [Soft labels]', color, n_bins=10,
                          show_gap_ann=(name == "TS"))
        if T is not None:
            ann_T(axes[0][col], T, 'hard cal'); ann_T(axes[1][col], T, 'soft cal')
        ea, _ = ece_bins(p[amb_te],  ys_te[amb_te])
        ec, _ = ece_bins(p[~amb_te], ys_te[~amb_te])
        eh_vals.append(eh); es_vals.append(es)
        ea_vals.append(ea); ec_vals.append(ec); names.append(name)

    axes[0][0].set_ylabel('Evaluated with\nHard (voted) labels\n\nAvg. label prob.', fontsize=8.5)
    axes[1][0].set_ylabel('Evaluated with\nSoft (annotator) labels\n\nAvg. label prob.', fontsize=8.5)

    # Annotation on TS soft-label diagram
    axes[1][1].annotate(
        'Calibration gap:\nTS overconfident\n(voted labels mislead TS)',
        xy=(0.80, 0.68), xytext=(0.35, 0.22), xycoords='data', textcoords='data',
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
        fontsize=8.5, color='darkred',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF0F0', ec='darkred', alpha=0.9))

    # Gap-shading legend on SLTS soft diagram
    axes[1][2].legend(
        handles=[mpatches.Patch(color=C['gap_over'],  alpha=0.6, label='Overconfident'),
                 mpatches.Patch(color=C['gap_under'], alpha=0.6, label='Underconfident')],
        fontsize=8, loc='upper left')

    # ---- Overall ECE bar chart ----
    x  = np.arange(3); w = 0.30
    cols3 = [C['uncal'], C['ts'], C['slts']]
    bh = ax_ov.bar(x-w/2, eh_vals, w, label='ECE-Hard (vs. voted labels)',
                   color=cols3, alpha=0.50, edgecolor='black', lw=0.7, hatch='//')
    bs = ax_ov.bar(x+w/2, es_vals, w, label='ECE-Soft (vs. annotator dist.)',
                   color=cols3, alpha=0.90, edgecolor='black', lw=0.8)
    val_labels(ax_ov, list(bh)+list(bs))

    ts_i = names.index('TS')
    gap  = es_vals[ts_i] - eh_vals[ts_i]
    ax_ov.annotate('', xy=(ts_i+w/2, es_vals[ts_i]), xytext=(ts_i-w/2, eh_vals[ts_i]),
                   arrowprops=dict(arrowstyle='<->', color='darkred', lw=2.2))
    ax_ov.text(ts_i+.06, (es_vals[ts_i]+eh_vals[ts_i])/2,
               f'Calibration gap\nΔ = {gap:+.3f}',
               color='darkred', fontsize=9, va='center', fontweight='bold')

    ax_ov.set_xticks(x); ax_ov.set_xticklabels(names, fontsize=11)
    ax_ov.set_ylabel('ECE', fontsize=10)
    ax_ov.set_title('Overall ECE: hard vs. soft labels\n'
                    'TS looks calibrated on hard labels — but the gap appears on soft labels',
                    fontsize=10, fontweight='bold')
    ax_ov.legend(fontsize=9); ax_ov.grid(True, alpha=0.3, axis='y')
    ax_ov.set_ylim(0, max(es_vals)*1.65)

    # ---- Stratified bar chart ----
    x2 = np.arange(2); w2 = 0.28
    for i, (nm, ea, ec, col) in enumerate([
            ("TS",          ea_vals[1], ec_vals[1], C['ts']),
            ("SLTS (ours)", ea_vals[2], ec_vals[2], C['slts'])]):
        off = (i-.5)*(w2+.04)
        b = ax_strat.bar(x2+off, [ea, ec], w2, label=nm, color=col,
                         alpha=0.85, edgecolor='black', lw=0.7)
        val_labels(ax_strat, b)

    ax_strat.set_xticks(x2)
    ax_strat.set_xticklabels(['Ambiguous\nsamples', 'Clear\nsamples'], fontsize=10)
    ax_strat.set_ylabel('ECE-Soft', fontsize=10)
    ax_strat.set_title('ECE-Soft stratified\nby ambiguity',
                       fontsize=10, fontweight='bold')
    ax_strat.legend(fontsize=9); ax_strat.grid(True, alpha=0.3, axis='y')
    ax_strat.set_ylim(0, max(ea_vals)*1.55)

    plt.savefig('toy_example_figure.png', dpi=150, bbox_inches='tight')
    print('\nFigure saved → toy_example_figure.png')
    plt.show()


if __name__ == '__main__':
    main()
