"""
Plot the data distribution of the toy example.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.stats import multivariate_normal

np.random.seed(42)

# ---- same parameters as run_toy_example.py ----
n_per_class = 300   # fewer points for cleaner scatter
cov     = np.diag([1.0, 0.5])
centers = [np.array([-3., 0.]), np.array([0., 0.]), np.array([3., 0.])]
PI_AMB  = [0., 0.7, 0.3]

X0 = np.random.multivariate_normal(centers[0], cov, n_per_class)
X1 = np.random.multivariate_normal(centers[1], cov, n_per_class)
X2 = np.random.multivariate_normal(centers[2], cov, n_per_class)

# voted hard labels (class 1 cluster always labeled "1")
y_hard = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

# For class 1, simulate what "one annotator's label" looks like
y_ann_class1 = np.random.choice([1, 2], size=n_per_class, p=[0.7, 0.3])

# -----------------------------------------------------------------------
# Figure: 3 panels
#   Left  : scatter of features colored by TRUE CLASS (cluster membership)
#   Middle: scatter colored by VOTED HARD LABEL (what the model sees)
#   Right : pie charts showing π for each class
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(15, 5.5))
fig.suptitle('Toy Example — Data Distribution', fontsize=14, fontweight='bold', y=1.02)

gs = fig.add_gridspec(1, 3, wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

COLORS = {0: '#5A8FC2', 1: '#E07040', 2: '#4CAF80'}
LABELS = {0: 'Class 0 (clear)', 1: 'Class 1 (ambiguous)', 2: 'Class 2 (clear)'}

# ---- Panel 1: colored by cluster membership (true class) ----
for i, X in enumerate([X0, X1, X2]):
    ax1.scatter(X[:, 0], X[:, 1], c=COLORS[i], alpha=0.45, s=18,
                label=LABELS[i], edgecolors='none')

# draw density contours
x_grid = np.linspace(-7, 7, 200)
y_grid = np.linspace(-2.5, 2.5, 100)
XX, YY = np.meshgrid(x_grid, y_grid)
pos = np.dstack([XX, YY])
for i, center in enumerate(centers):
    rv  = multivariate_normal(center, cov)
    Z   = rv.pdf(pos)
    ax1.contour(XX, YY, Z, levels=4, colors=[COLORS[i]],
                alpha=0.6, linewidths=1.0)

ax1.set_title('Feature space\n(colored by cluster / true class)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Feature 1', fontsize=10); ax1.set_ylabel('Feature 2', fontsize=10)
ax1.legend(fontsize=8.5, loc='upper right')
ax1.set_xlim(-7, 7); ax1.set_ylim(-2.5, 2.5)
ax1.grid(True, alpha=0.2)
ax1.axvline(x=0, color='gray', lw=0.5, ls='--')

# ---- Panel 2: colored by VOTED hard label ----
# Class 0 and 2: voted = true class (no change)
# Class 1: ALL samples colored as label=1 (voted), but annotator sometimes says 2
marker_voted  = dict(alpha=0.55, s=18, edgecolors='none')
marker_rogue  = dict(alpha=0.85, s=30, edgecolors='black', linewidths=0.4)

ax2.scatter(X0[:, 0], X0[:, 1], c=COLORS[0], **marker_voted)
ax2.scatter(X2[:, 0], X2[:, 1], c=COLORS[2], **marker_voted)

# class 1 cluster: split by what a SINGLE annotator would say
mask_1 = (y_ann_class1 == 1)
mask_2 = (y_ann_class1 == 2)
ax2.scatter(X1[mask_1, 0], X1[mask_1, 1], c=COLORS[1], **marker_voted,
            label=f'Annotator says "class 1" ({mask_1.sum()}/{n_per_class} ≈ 70%)')
ax2.scatter(X1[mask_2, 0], X1[mask_2, 1], c=COLORS[2], marker='D', **marker_rogue,
            label=f'Annotator says "class 2" ({mask_2.sum()}/{n_per_class} ≈ 30%)')

# box highlighting the ambiguous cluster
from matplotlib.patches import FancyBboxPatch
bbox = FancyBboxPatch((-2.8, -2.3), 5.6, 4.6,
                       boxstyle='round,pad=0.1', linewidth=1.5,
                       edgecolor='#E07040', facecolor='#FFF5EE', alpha=0.25, zorder=0)
ax2.add_patch(bbox)
ax2.text(0, 2.15, 'Ambiguous cluster\n(same features, different labels)',
         ha='center', fontsize=8, color='#E07040',
         bbox=dict(fc='white', ec='#E07040', alpha=0.8, pad=2))

ax2.set_title('What the model sees\n(colored by single annotator label)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Feature 1', fontsize=10); ax2.set_ylabel('Feature 2', fontsize=10)
ax2.legend(fontsize=8, loc='lower right')
ax2.set_xlim(-7, 7); ax2.set_ylim(-2.5, 2.5)
ax2.grid(True, alpha=0.2)
ax2.axvline(x=0, color='gray', lw=0.5, ls='--')

# voted label annotation
ax2.annotate('Voted (majority) label\nalways = class 1',
             xy=(0.5, -1.8), xytext=(2.5, -1.5),
             arrowprops=dict(arrowstyle='->', color='darkred', lw=1.3),
             fontsize=8, color='darkred',
             bbox=dict(fc='#FFF0F0', ec='darkred', pad=2, alpha=0.9))

# ---- Panel 3: π for each class (pie charts + diagram) ----
ax3.axis('off')

# Draw 3 pie charts stacked vertically
pie_ax = [fig.add_axes([0.70 + 0.01, 0.62 - k*0.28, 0.09, 0.22]) for k in range(3)]

pie_data = [
    ([1.0],        [''],        [COLORS[0]],            'Class 0 (clear)\nπ = [1.0, 0.0, 0.0]'),
    ([0.7, 0.3],   ['70%','30%'], [COLORS[1], COLORS[2]], 'Class 1 (ambiguous)\nπ = [0.0, 0.7, 0.3]'),
    ([1.0],        [''],        [COLORS[2]],            'Class 2 (clear)\nπ = [0.0, 0.0, 1.0]'),
]

for k, (sizes, lbls, cols, title) in enumerate(pie_data):
    pie_ax[k].pie(sizes, labels=lbls, colors=cols, autopct='' if len(sizes)==1 else None,
                  startangle=90, wedgeprops=dict(linewidth=0.8, edgecolor='white'))
    pie_ax[k].set_title(title, fontsize=8.5, fontweight='bold', pad=4)

# Text explanation on ax3
ax3.text(0.0, 0.97,
    'Annotator label distribution  π(·|x)',
    fontsize=11, fontweight='bold', va='top', transform=ax3.transAxes)

ax3.text(0.0, 0.88,
    'Each class has a fixed distribution\n'
    'over labels that annotators assign.\n\n'
    'Clear classes: all annotators agree.\n'
    'Ambiguous class: 70% say "class 1",\n'
    '                 30% say "class 2".\n\n'
    '→ Voted label = always class 1\n'
    '  (hides the 30% disagreement)\n\n'
    '→ Soft label = π = [0, 0.7, 0.3]\n'
    '  (preserves the disagreement)\n\n'
    '→ Features X cannot distinguish\n'
    '  which annotator is "right" —\n'
    '  all class 1 points look the same.',
    fontsize=9, va='top', transform=ax3.transAxes,
    bbox=dict(fc='#F8F8F8', ec='gray', pad=8, alpha=0.7))

plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
print('Saved → data_distribution.png')
plt.show()
