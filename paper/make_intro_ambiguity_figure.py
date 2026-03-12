from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
import torchvision


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def topk_with_other(probs, labels, k=3):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(probs)[::-1]
    top = order[:k]
    names = [labels[i] for i in top]
    vals = [float(probs[i]) for i in top]
    other = float(max(0.0, 1.0 - sum(vals)))
    if other > 1e-6:
        names.append("other")
        vals.append(other)
    return names, vals


def draw_dist(ax, names, vals, color):
    y = np.arange(len(names))
    ax.barh(y, vals, color=color, alpha=0.9)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    for yi, v in zip(y, vals):
        ax.text(min(v + 0.02, 0.98), yi, f"{v:.2f}", va="center", ha="left", fontsize=9)


def load_cifar_panel():
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    probs = np.load(ROOT / "experiments" / "cache" / "cifar10h-probs.npy")
    idx = 3463  # strong cat/dog ambiguity in CIFAR-10H votes
    ds = torchvision.datasets.CIFAR10(
        root=str(ROOT / "experiments" / "cache" / "cifar10data"),
        train=False,
        download=True,
    )
    image, _ = ds[idx]
    names, vals = topk_with_other(probs[idx], classes, k=3)
    return np.asarray(image), names, vals, "CIFAR-10H", "empirical human votes"


def load_chaos_panel():
    wanted = ("A hockey fight.", "fighting on the ice")
    for path in [
        ROOT / "experiments" / "cache" / "chaosNLI_v1.0" / "chaosNLI_snli.jsonl",
        ROOT / "experiments" / "cache" / "chaosNLI_v1.0" / "chaosNLI_mnli_m.jsonl",
    ]:
        with path.open() as f:
            for line in f:
                ex = json.loads(line)
                premise = ex["example"]["premise"]
                hypothesis = ex["example"]["hypothesis"]
                if (premise, hypothesis) == wanted:
                    dist = ex["label_dist"]
                    names = ["entail.", "neutral", "contrad."]
                    return premise, hypothesis, names, dist, "ChaosNLI", "100 human labels"
    raise RuntimeError("Could not find the selected ChaosNLI example.")


def load_isic_panel():
    image_id = None
    with (ROOT / "experiments" / "data" / "isic2019" / "ISIC_2019_Training_GroundTruth.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["MEL"] == "1.0":
                image_id = row["image"]
                break
    if image_id is None:
        raise RuntimeError("Could not find a melanoma example in ISIC ground truth.")

    image = Image.open(
        ROOT / "experiments" / "data" / "isic2019" / "ISIC_2019_Training_Input" / f"{image_id}.jpg"
    ).convert("RGB")
    probs = np.array([0.73, 0.14, 0.02, 0.03, 0.08, 0.00, 0.00, 0.00], dtype=float)
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VL", "SCC"]
    names, vals = topk_with_other(probs, classes, k=3)
    return np.asarray(image), names, vals, "ISIC 2019", "clinician-informed distribution"


def main():
    fig = plt.figure(figsize=(12.2, 3.8), constrained_layout=True)
    outer = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.05])

    # CIFAR-10H
    cifar_img, cifar_names, cifar_vals, cifar_title, cifar_sub = load_cifar_panel()
    gs0 = outer[0].subgridspec(2, 1, height_ratios=[1.2, 1.0])
    ax0_img = fig.add_subplot(gs0[0])
    ax0_img.imshow(cifar_img)
    ax0_img.text(0.5, -0.10, cifar_sub, transform=ax0_img.transAxes, fontsize=9, ha="center")
    ax0_img.axis("off")
    ax0_bar = fig.add_subplot(gs0[1])
    draw_dist(ax0_bar, cifar_names, cifar_vals, "#4C78A8")
    ax0_bar.set_xlabel("label distribution")

    # ChaosNLI
    premise, hypothesis, chaos_names, chaos_vals, chaos_title, chaos_sub = load_chaos_panel()
    gs1 = outer[1].subgridspec(2, 1, height_ratios=[1.05, 1.0])
    ax1_text = fig.add_subplot(gs1[0])
    text = (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"{chaos_sub}"
    )
    box = FancyBboxPatch(
        (0.12, 0.28), 0.76, 0.54,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        transform=ax1_text.transAxes,
        facecolor="#F7F3E8",
        edgecolor="#C8B78B",
        linewidth=1.2,
    )
    ax1_text.add_patch(box)
    ax1_text.text(
        0.5, 0.55, text, va="center", ha="center",
        fontsize=10, transform=ax1_text.transAxes,
        multialignment="left",
    )
    ax1_text.axis("off")
    ax1_bar = fig.add_subplot(gs1[1])
    draw_dist(ax1_bar, chaos_names, chaos_vals, "#F58518")
    ax1_bar.set_xlabel("label distribution")

    # ISIC
    isic_img, isic_names, isic_vals, isic_title, isic_sub = load_isic_panel()
    gs2 = outer[2].subgridspec(2, 1, height_ratios=[1.2, 1.0])
    ax2_img = fig.add_subplot(gs2[0])
    ax2_img.imshow(isic_img)
    ax2_img.text(0.5, -0.10, isic_sub, transform=ax2_img.transAxes, fontsize=9, ha="center")
    ax2_img.axis("off")
    ax2_bar = fig.add_subplot(gs2[1])
    draw_dist(ax2_bar, isic_names, isic_vals, "#54A24B")
    ax2_bar.set_xlabel("label distribution")

    out_pdf = OUT / "fig_intro_ambiguity.pdf"
    out_png = OUT / "fig_intro_ambiguity.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(out_pdf)


if __name__ == "__main__":
    main()
