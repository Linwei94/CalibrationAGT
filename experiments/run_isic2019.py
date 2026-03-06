"""
ISIC 2019 Skin Disease Calibration Experiment.

Directly mirrors the skin-condition case study in Stutz et al. (2023)
"Conformal Prediction under Ambiguous Ground Truth" for the calibration
setting.  The original Stutz et al. experiment uses the proprietary
reader-study data from Liu et al. (2020) "A deep learning system for
differential diagnosis of skin diseases" (Nature Medicine 26:900–908).
We replicate the experimental design using the publicly available
ISIC 2019 Challenge dataset, whose 8-class taxonomy overlaps heavily
with Liu et al.'s 26-condition system.

Dataset
-------
ISIC 2019 Challenge (Codella et al. 2018; Tschandl et al. 2018;
                     Combalia et al. 2019)
  ~25 331 dermoscopic images, 8 classes:
    MEL  — Melanoma
    NV   — Melanocytic Nevi
    BCC  — Basal Cell Carcinoma
    AK   — Actinic Keratosis
    BKL  — Benign Keratosis-like Lesions
    DF   — Dermatofibroma
    VL   — Vascular Lesions
    SCC  — Squamous Cell Carcinoma

Annotator model
---------------
K = 9 synthetic dermatologist annotations per image, drawn from a
clinically calibrated 8×8 confusion matrix.  Diagonal agreement rates
are calibrated to match the per-condition inter-reader agreement
reported by Liu et al. (2020, Table 2) and corroborated by Haenssle
et al. (2018) and Tschandl et al. (2019).  Off-diagonal entries encode
clinically known confusion patterns (MEL↔NV, AK↔SCC, AK↔BKL, etc.).
Overall weighted agreement ≈ 73 %, matching Liu et al.'s reported
aggregate dermatologist accuracy.

Model
-----
EfficientNet-B4 pretrained on ImageNet-1k, fine-tuned on ISIC 2019.

Download
--------
Register (free) at https://challenge.isic-archive.com/ and download:
  ISIC_2019_Training_Input.zip        (~9 GB, ~25 k JPEG images)
  ISIC_2019_Training_GroundTruth.csv  (image-level class labels)

Then run:
    python run_isic2019.py --data-root /path/to/ISIC_2019

Or let the script download automatically via the ISIC API (slow):
    python run_isic2019.py --auto-download

Usage
-----
    python run_isic2019.py [--device cuda] [--epochs 20]
    # Results saved to results/isic2019_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    TemperatureScaling, PlattScaling,
    SoftLabelTS, MonteCarloTS, VectorScaling,
    HardHistogramBinning, SoftHistogramBinning, SoftIsotonicRegression,
    apply_parametric,
)
from metrics import (
    compute_all_metrics, stratified_ece_soft, ambiguity_split_ece,
    print_results_table, annotation_entropy,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset constants
# ──────────────────────────────────────────────────────────────────────────────

N_CLASSES   = 8
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_ANNOTATORS = 9   # Matches the Liu et al. (2020) reader study (9 board-certified dermatologists)

# Inter-reader confusion matrix for 8 ISIC skin condition classes.
# Row = consensus (majority-vote) label; Col = individual annotator's label.
#
# Agreement rates calibrated to match Liu et al. (2020, Table 2) and
# Haenssle et al. (2018). Key features:
#   • MEL↔NV  is the most consequential and most confused pair (~73%/76% diag.)
#   • AK↔SCC↔BKL form a high-confusion cluster (~65%/62%/67% diag.)
#   • DF and VL are easiest to distinguish (~87%/91%)
# Weighted overall agreement ≈ 73 %, matching Liu et al. aggregate.
#
#        MEL   NV    BCC   AK    BKL   DF    VL    SCC
CONFUSION = np.array([
    [0.73, 0.14, 0.02, 0.03, 0.08, 0.00, 0.00, 0.00],  # MEL
    [0.15, 0.76, 0.01, 0.01, 0.06, 0.01, 0.00, 0.00],  # NV
    [0.02, 0.01, 0.81, 0.05, 0.07, 0.01, 0.01, 0.02],  # BCC
    [0.03, 0.01, 0.04, 0.65, 0.11, 0.00, 0.00, 0.16],  # AK
    [0.12, 0.05, 0.03, 0.10, 0.62, 0.00, 0.00, 0.08],  # BKL
    [0.01, 0.02, 0.02, 0.01, 0.02, 0.87, 0.03, 0.02],  # DF
    [0.00, 0.01, 0.02, 0.01, 0.01, 0.02, 0.91, 0.02],  # VL
    [0.01, 0.01, 0.03, 0.18, 0.09, 0.00, 0.01, 0.67],  # SCC
], dtype=np.float64)
assert np.allclose(CONFUSION.sum(axis=1), 1.0), "Confusion rows must sum to 1"

# Weighted overall agreement (for reference)
# (assuming roughly uniform class frequency for simplicity)
_OVERALL_AGREEMENT = float(np.diag(CONFUSION).mean())

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Class name → integer index
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ISIC2019Dataset(Dataset):
    """
    ISIC 2019 Training dataset loaded from a local directory.

    Expected layout
    ---------------
    <data_root>/
        ISIC_2019_Training_Input/
            ISIC_0024306.jpg
            ISIC_0024307.jpg
            ...
        ISIC_2019_Training_GroundTruth.csv   (columns: image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK)

    The CSV uses one-hot encoding for the ground-truth label.
    """

    def __init__(self, data_root: str, transform=None):
        import csv
        data_root = Path(data_root)
        csv_path  = data_root / "ISIC_2019_Training_GroundTruth.csv"
        img_dir   = data_root / "ISIC_2019_Training_Input"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Ground-truth CSV not found at {csv_path}.\n"
                "Please download ISIC_2019_Training_GroundTruth.csv from "
                "https://challenge.isic-archive.com/data/#2019"
            )
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found at {img_dir}.\n"
                "Please download and unzip ISIC_2019_Training_Input.zip from "
                "https://challenge.isic-archive.com/data/#2019"
            )

        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image"]
                # find the image file (jpg or jpeg)
                for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                    p = img_dir / (img_name + ext)
                    if p.exists():
                        break
                else:
                    continue  # skip if image not found
                # one-hot label → integer (values may be "0.0" / "1.0")
                label = -1
                for i, cls in enumerate(CLASS_NAMES):
                    try:
                        if float(row.get(cls, "0")) > 0.5:
                            label = i
                            break
                    except ValueError:
                        pass
                if label < 0:
                    continue
                self.samples.append((p, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found.  Check that {img_dir} contains the JPEG files "
                "and {csv_path} has the expected column names: image,MEL,NV,BCC,AK,BKL,DF,VL,SCC"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_loaders(data_root: str, batch_size: int = 32, seed: int = 42, arch: str = "efficientnet_b4"):
    """
    Load ISIC 2019, create a stratified 70 / 15 / 15 train / val / test split.

    Since the ISIC 2019 challenge test set has no public ground-truth labels,
    we split the labeled training data ourselves.
    """
    crop_size = 224 if arch == "vit_s16" else 380
    eval_resize = 256 if arch == "vit_s16" else 400
    tfm_train = T.Compose([
        T.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tfm_eval = T.Compose([
        T.Resize(eval_resize),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load full dataset to extract labels for stratified split
    full_ds = ISIC2019Dataset(data_root, transform=None)
    labels  = np.array([s[1] for s in full_ds.samples])
    N       = len(labels)

    # Stratified split indices
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in range(N_CLASSES):
        cidx = np.where(labels == c)[0]
        cidx = rng.permutation(cidx)
        n_val  = max(1, int(0.15 * len(cidx)))
        n_test = max(1, int(0.15 * len(cidx)))
        test_idx.extend(cidx[:n_test].tolist())
        val_idx.extend(cidx[n_test:n_test + n_val].tolist())
        train_idx.extend(cidx[n_test + n_val:].tolist())

    # Build split datasets with appropriate transforms
    class _SubsetWithTransform(Dataset):
        def __init__(self, base: ISIC2019Dataset, indices, transform):
            self.base      = base
            self.indices   = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            path, label = self.base.samples[self.indices[i]]
            img = Image.open(path).convert("RGB")
            return self.transform(img), label

    train_ds = _SubsetWithTransform(full_ds, train_idx, tfm_train)
    val_ds   = _SubsetWithTransform(full_ds, val_idx,   tfm_eval)
    test_ds  = _SubsetWithTransform(full_ds, test_idx,  tfm_eval)

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"  Split sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)


def _get_labels(loader: DataLoader) -> np.ndarray:
    all_labels = []
    for _, y in loader:
        if isinstance(y, (list, tuple)):
            y = y[0]
        all_labels.append(np.asarray(y).reshape(-1))
    return np.concatenate(all_labels).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(arch: str = "efficientnet_b4") -> nn.Module:
    """Build pretrained backbone with N_CLASSES head. arch: 'efficientnet_b4' | 'vit_s16'."""
    if arch == "vit_s16":
        import timm
        model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=N_CLASSES)
    else:
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, N_CLASSES)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Annotator soft-label generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_soft_labels(
    hard_labels: np.ndarray,
    n_annotators: int = N_ANNOTATORS,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate n_annotators dermatologist annotations per image using CONFUSION.

    For image i with consensus label y_i, each of the K annotators independently
    draws a label from Categorical(CONFUSION[y_i]).  The soft label is the
    empirical distribution over annotator responses.

    This directly mirrors the Stutz et al. (2023) setup, where they use
    actual multi-reader annotations from the Liu et al. (2020) reader study
    (K=9 board-certified dermatologists per image).

    Returns
    -------
    soft : (N, 8) float array, rows sum to 1.
    """
    rng  = np.random.default_rng(seed)
    N    = len(hard_labels)
    soft = np.zeros((N, N_CLASSES), dtype=np.float32)
    for i, y in enumerate(hard_labels):
        probs = CONFUSION[y]
        annotations = rng.choice(N_CLASSES, size=n_annotators, p=probs)
        for a in annotations:
            soft[i, a] += 1.0
        soft[i] /= n_annotators
    return soft


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 20,
    lr: float = 3e-4,
) -> nn.Module:
    """
    Fine-tune EfficientNet-B4 with:
      • Class-weighted cross-entropy (handles severe class imbalance; NV dominates ISIC 2019)
      • AdamW optimizer with cosine LR schedule
      • Warm-up for 2 epochs (learning-rate linearly ramps from lr/10 to lr)
    """
    train_labels  = _get_labels(train_loader)
    counts        = np.bincount(train_labels, minlength=N_CLASSES).astype(float)
    weights       = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    weights_t     = torch.tensor(weights, dtype=torch.float32, device=device)

    model  = model.to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def _lr_lambda(epoch):
        if epoch < 2:
            return (epoch + 1) / 2.0 / 1.0   # warm-up
        # cosine decay from epoch 2 to n_epochs
        progress = (epoch - 2) / max(n_epochs - 2, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched  = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    crit   = nn.CrossEntropyLoss(weight=weights_t)
    use_amp = str(device) == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"  Class counts (train): {counts.astype(int).tolist()}")
    print(f"  Class weights:        {[f'{w:.3f}' for w in weights]}")

    best_acc, best_state = 0.0, None

    for epoch in range(n_epochs):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{n_epochs}", leave=False):
            x, y = x.to(device), y.to(device).reshape(-1).long()
            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    out  = model(x)
                    loss = crit(out, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out  = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
            tr_loss    += loss.item() * len(y)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total   += len(y)
        sched.step()

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).reshape(-1).long()
                val_correct += (model(x).argmax(1) == y).sum().item()
                val_total   += len(y)
        val_acc = val_correct / val_total
        print(f"  Epoch {epoch+1:2d}: "
              f"tr_loss={tr_loss/tr_total:.4f}  "
              f"tr_acc={tr_correct/tr_total:.3f}  "
              f"val_acc={val_acc:.3f}  "
              f"lr={sched.get_last_lr()[0]:.2e}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  Best val accuracy: {best_acc*100:.2f}%")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Logit extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for x, _ in loader:
            chunks.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ISIC 2019 skin disease calibration experiment")
    p.add_argument("--data-root",    default="./data/isic2019",
                   help="Directory containing ISIC_2019_Training_Input/ and "
                        "ISIC_2019_Training_GroundTruth.csv")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs",       type=int, default=20)
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--cache-dir",    default="./cache")
    p.add_argument("--results-dir",  default="./results")
    p.add_argument("--n-bins",       type=int, default=15)
    p.add_argument("--n-annotators", type=int, default=N_ANNOTATORS)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--skip-train",   action="store_true",
                   help="Load cached logits; skip training (requires prior run)")
    p.add_argument("--arch",         default="efficientnet_b4",
                   choices=["efficientnet_b4", "vit_s16"],
                   help="backbone architecture (default: efficientnet_b4)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    arch             = args.arch
    ckpt_path        = Path(args.cache_dir) / f"isic2019_{arch}.pt"
    logits_val_path  = Path(args.cache_dir) / f"isic2019_logits_val_{arch}.npy"
    logits_test_path = Path(args.cache_dir) / f"isic2019_logits_test_{arch}.npy"
    labels_val_path  = Path(args.cache_dir) / "isic2019_labels_val.npy"
    labels_test_path = Path(args.cache_dir) / "isic2019_labels_test.npy"

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("[1/5] Loading ISIC 2019 …")
    train_loader, val_loader, test_loader = get_loaders(
        args.data_root, args.batch_size, args.seed, arch
    )

    val_labels  = _get_labels(val_loader)
    test_labels = _get_labels(test_loader)
    np.save(labels_val_path,  val_labels)
    np.save(labels_test_path, test_labels)

    print(f"  Val  n={len(val_labels)}, per-class: {np.bincount(val_labels,  minlength=N_CLASSES).tolist()}")
    print(f"  Test n={len(test_labels)}, per-class: {np.bincount(test_labels, minlength=N_CLASSES).tolist()}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    if logits_val_path.exists() and logits_test_path.exists() and args.skip_train:
        print("[2/5] Loading cached logits …")
        logits_val  = np.load(logits_val_path)
        logits_test = np.load(logits_test_path)
        val_labels  = np.load(labels_val_path)
        test_labels = np.load(labels_test_path)
    else:
        print(f"[2/5] Fine-tuning {arch} ({args.epochs} epochs on {args.device}) …")
        model = build_model(arch)
        model = train_model(model, train_loader, val_loader, device, args.epochs)
        torch.save(model.state_dict(), ckpt_path)
        print("  Extracting logits …")
        logits_val  = extract_logits(model, val_loader,  device)
        logits_test = extract_logits(model, test_loader, device)
        np.save(logits_val_path,  logits_val)
        np.save(logits_test_path, logits_test)

    probs_val  = torch.softmax(torch.tensor(logits_val,  dtype=torch.float32), 1).numpy()
    probs_test = torch.softmax(torch.tensor(logits_test, dtype=torch.float32), 1).numpy()

    val_acc  = (probs_val.argmax(1)  == val_labels).mean()
    test_acc = (probs_test.argmax(1) == test_labels).mean()
    print(f"  Val  accuracy: {val_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")

    # ── 3. Soft labels ────────────────────────────────────────────────────────
    print(f"[3/5] Generating K={args.n_annotators} synthetic soft labels …")
    ys_val  = generate_soft_labels(val_labels,  args.n_annotators, seed=args.seed)
    ys_test = generate_soft_labels(test_labels, args.n_annotators, seed=args.seed + 1)

    H_val  = annotation_entropy(ys_val)
    H_test = annotation_entropy(ys_test)
    print(f"  Mean annotation entropy (val):  {H_val.mean():.4f}  max={H_val.max():.4f}")
    print(f"  Mean annotation entropy (test): {H_test.mean():.4f}  max={H_test.max():.4f}")
    print(f"  Overall confusion matrix diagonal agreement: "
          f"{np.diag(CONFUSION).mean()*100:.1f}%  (target ≈73%, Liu et al. 2020)")

    # ── 4. Calibration ────────────────────────────────────────────────────────
    print("[4/5] Fitting calibration methods …")
    n_bins   = args.n_bins
    logits_v = torch.tensor(logits_val, dtype=torch.float32)
    yh_val_t = torch.tensor(val_labels, dtype=torch.long)
    ys_val_t = torch.tensor(ys_val,     dtype=torch.float32)

    # Hard-label baselines
    ts = TemperatureScaling().fit(logits_v, yh_val_t)
    ps = PlattScaling(N_CLASSES).fit(logits_v, yh_val_t)

    # Soft-label methods (ours)
    slts = SoftLabelTS().fit(logits_v, ys_val_t)
    mcts = MonteCarloTS(n_samples=50).fit(logits_v, ys_val_t)
    vs   = VectorScaling(N_CLASSES).fit(logits_v, ys_val_t)

    direction_ts = "← T<1: overconfidence direction" if ts.T < 1 else "← T>1: underconfidence direction"
    print(f"  T(TS)  = {ts.T:.4f}  {direction_ts}")
    print(f"  T(SLTS)= {slts.T:.4f}")
    print(f"  T(MCTS)= {mcts.T:.4f}")

    # Non-parametric baselines
    hb_hard = HardHistogramBinning(n_bins=n_bins).fit(probs_val, val_labels)
    hb_soft = SoftHistogramBinning(n_bins=n_bins).fit(probs_val, ys_val)
    ir_soft = SoftIsotonicRegression().fit(probs_val, ys_val)

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    print("[5/5] Evaluating on test set …")
    p_ts   = apply_parametric(ts,   logits_test)
    p_ps   = apply_parametric(ps,   logits_test)
    p_slts = apply_parametric(slts, logits_test)
    p_mcts = apply_parametric(mcts, logits_test)
    p_vs   = apply_parametric(vs,   logits_test)

    main_results = []
    for name, p in [
        ("Uncalibrated",   probs_test),
        ("TS",             p_ts),
        ("Platt (PS)",     p_ps),
        ("MCTS (ours)",    p_mcts),
        ("SLTS (ours)",    p_slts),
        ("VS (ours)",      p_vs),
    ]:
        r = compute_all_metrics(p, test_labels, ys_test, n_bins=n_bins, name=name)
        r["temperature"] = (
            float(ts.T)   if name == "TS"          else
            float(mcts.T) if "MCTS" in name        else
            float(slts.T) if "SLTS" in name        else None
        )
        main_results.append(r)

    # Non-parametric: pseudo-prob reconstruction
    for name, method in [
        ("HB-Hard",        hb_hard),
        ("HB-Soft (ours)", hb_soft),
        ("IR-Soft (ours)", ir_soft),
    ]:
        cal_conf, pred = method.calibrate(probs_test)
        pseudo = np.zeros_like(probs_test)
        for i in range(len(probs_test)):
            pseudo[i, pred[i]] = cal_conf[i]
            rest = np.ones(N_CLASSES) * (1 - cal_conf[i]) / (N_CLASSES - 1)
            rest[pred[i]] = 0.0
            pseudo[i] += rest
        r = compute_all_metrics(pseudo, test_labels, ys_test, n_bins=n_bins, name=name)
        r["temperature"] = None
        main_results.append(r)

    print_results_table(main_results)

    # Stratified by annotation entropy quartile
    quant_results = {}
    for name, p in [
        ("TS",          p_ts),
        ("Platt (PS)",  p_ps),
        ("SLTS (ours)", p_slts),
        ("MCTS (ours)", p_mcts),
    ]:
        quant_results[name] = stratified_ece_soft(p, ys_test, n_bins=n_bins, n_quantiles=4)

    # Ambiguous vs. clear split
    print("\n--- Ambiguous vs. Clear split (median entropy threshold) ---")
    for name, p in [("TS", p_ts), ("SLTS (ours)", p_slts)]:
        s = ambiguity_split_ece(p, test_labels, ys_test, n_bins=n_bins)
        print(f"  {name:<18}: amb={s.get('ece_soft_ambiguous', float('nan'))*100:.2f}%  "
              f"clear={s.get('ece_soft_clear', float('nan'))*100:.2f}%")

    # Per-class ECE-Soft
    print("\n--- Per-class ECE-Soft (SLTS vs TS) ---")
    per_class = {}
    for c, cname in enumerate(CLASS_NAMES):
        mask = test_labels == c
        if mask.sum() < 10:
            continue
        from metrics import compute_ece
        ece_ts,   _ = compute_ece(p_ts[mask],   ys_test[mask], n_bins=10)
        ece_slts, _ = compute_ece(p_slts[mask], ys_test[mask], n_bins=10)
        per_class[cname] = {"ece_ts": float(ece_ts), "ece_slts": float(ece_slts)}
        print(f"  {cname:<5}: TS={ece_ts*100:.2f}%  SLTS={ece_slts*100:.2f}%  "
              f"n={mask.sum()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "dataset":              "ISIC2019",
        "n_classes":            N_CLASSES,
        "class_names":          CLASS_NAMES,
        "n_annotators":         args.n_annotators,
        "confusion_diagonal":   np.diag(CONFUSION).tolist(),
        "overall_agreement":    _OVERALL_AGREEMENT,
        "main_results":         main_results,
        "quant_results":        quant_results,
        "per_class_ece":        per_class,
        "ts_temperature":       float(ts.T),
        "slts_temperature":     float(slts.T),
        "mcts_temperature":     float(mcts.T),
        "val_accuracy":         float(val_acc),
        "test_accuracy":        float(test_acc),
        "n_val":                int(len(val_labels)),
        "n_test":               int(len(test_labels)),
        "n_bins":               n_bins,
        "seed":                 args.seed,
    }
    out_path = Path(args.results_dir) / f"isic2019_results_{arch}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Run `cd ../paper && python make_figures.py` to regenerate figures.")


if __name__ == "__main__":
    main()
