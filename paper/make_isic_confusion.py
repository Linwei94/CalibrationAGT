"""
Generate ISIC 2019 inter-reader annotator confusion matrix heatmap.

Outputs
-------
  figs/isic_confusion.pdf
  figs/isic_confusion.png

Usage
-----
    cd paper && python make_isic_confusion.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       13,
    "axes.labelsize":  13,
    "figure.dpi":      180,
    "pdf.fonttype":    42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── ISIC 2019 annotator confusion matrix ──────────────────────────────────────
# Entry C[i,j] = P(annotator says j | consensus label is i)
# Calibrated to match Liu et al. (2020) Table 2; mean diagonal = 75%
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

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VL", "SCC"]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 7))

cmap = mcolors.LinearSegmentedColormap.from_list("wblue", ["#ffffff", "#1a5fa8"], N=256)
im = ax.imshow(CONFUSION_ISIC, cmap=cmap, vmin=0, vmax=1, aspect="auto")

# Annotate each cell
for i in range(8):
    for j in range(8):
        v = CONFUSION_ISIC[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=10, color="white" if v > 0.45 else "black",
                fontweight="bold" if i == j else "normal")

ax.set_xticks(range(8));       ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_yticks(range(8));       ax.set_yticklabels(CLASS_NAMES, fontsize=11)
ax.set_xlabel("Annotator's label", fontsize=12)
ax.set_ylabel("Consensus (majority-vote) label", fontsize=12)

# Highlight MEL/NV confusion cluster
ax.add_patch(Rectangle((-0.5, -0.5), 2, 2,
                        fill=False, edgecolor="#E07040", lw=2.2, linestyle="--"))
ax.text(0.5, -0.78, "MEL/NV", ha="center", va="top",
        fontsize=9, color="#E07040", fontweight="bold")

# Highlight AK/BKL/SCC high-confusion cluster
for (ri, ci) in [(3, 7), (4, 7), (3, 4), (7, 3), (7, 4), (4, 3)]:
    ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                            fill=False, edgecolor="#9B59B6", lw=1.8, linestyle=":"))
ax.text(5.5, 7.78, "AK/BKL/SCC cluster", ha="center", va="bottom",
        fontsize=9, color="#9B59B6", fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Probability", fontsize=11)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(OUT / "isic_confusion.pdf", bbox_inches="tight")
plt.savefig(OUT / "isic_confusion.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved → figs/isic_confusion.pdf")
