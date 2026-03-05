"""
Main experiment: Calibration under Ambiguous Ground Truth on CIFAR-10H.

Pipeline
--------
1. Download CIFAR-10H soft annotations (10 000 × 10 human-vote probability matrix).
2. Load CIFAR-10 test images and a pretrained ResNet-50 fine-tuned on CIFAR-10.
3. Cache model logits on disk (avoids re-running the forward pass repeatedly).
4. Split CIFAR-10 test set into calibration (5 000) and test (5 000) sets.
5. Fit all calibration methods on the calibration split.
6. Evaluate hard + soft metrics on the test split.
7. Save results to experiments/results/cifar10h_results.json.

Usage
-----
    python run_cifar10h.py [--cache-dir ./cache] [--results-dir ./results]
                           [--n-bins 15] [--seed 42] [--device cpu|cuda]
                           [--epochs 30]   # only used when training from scratch

Requirements
------------
    torch torchvision timm requests scikit-learn tqdm
    pip install torch torchvision timm requests scikit-learn tqdm
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

# ── local imports ──────────────────────────────────────────────────────────────
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
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CIFAR10H_URL = (
    "https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy"
)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
N_CLASSES = 10


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def download_cifar10h(cache_dir: str) -> np.ndarray:
    """Download (once) and return the CIFAR-10H soft label matrix (10000, 10)."""
    path = Path(cache_dir) / "cifar10h-probs.npy"
    if not path.exists():
        print(f"Downloading CIFAR-10H annotations → {path}")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CIFAR10H_URL, path)
    probs = np.load(path).astype(np.float32)
    assert probs.shape == (10000, 10), f"Unexpected shape: {probs.shape}"
    # Renormalise rows (should sum to 1, but floating point)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def get_cifar10_testset(data_root: str):
    """Return the standard CIFAR-10 test dataset (10 000 images)."""
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(device: str, arch: str = "resnet50") -> nn.Module:
    """
    Build a pretrained ImageNet backbone fine-tuned for CIFAR-10 (10 classes).
    arch: 'resnet50' | 'vit_b16'
    """
    if arch == "vit_b16":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=N_CLASSES)
    else:
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def fine_tune_on_cifar10(
    model: nn.Module,
    data_root: str,
    device: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-4,
    checkpoint_path: str = None,
) -> nn.Module:
    """Fine-tune ResNet-50 on CIFAR-10 training set."""
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading fine-tuned checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model

    transform_train = T.Compose([
        T.Resize(224),
        T.RandomCrop(224, padding=28),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = (device == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device, scaler)
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{epochs}  loss={loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved → {checkpoint_path}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Logit extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_logits(
    model: nn.Module,
    dataset,
    device: str,
    batch_size: int = 256,
    cache_path: str = None,
) -> np.ndarray:
    """Run forward pass and return logits (N, K). Caches result to disk."""
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached logits: {cache_path}")
        return np.load(cache_path)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="  extracting logits"):
            logits = model(x.to(device)).cpu().numpy()
            all_logits.append(logits)

    logits = np.concatenate(all_logits, axis=0)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, logits)
        print(f"Logits cached → {cache_path}")
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    cache_dir   = args.cache_dir
    results_dir = args.results_dir
    device      = args.device
    n_bins      = args.n_bins
    seed        = args.seed
    epochs      = args.epochs
    arch        = args.arch

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data …")
    cifar10h  = download_cifar10h(cache_dir)                    # (10000, 10)
    testset   = get_cifar10_testset(os.path.join(cache_dir, "cifar10"))
    hard_labels = np.array([y for _, y in testset])            # (10000,)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print(f"\n[2/6] Preparing model ({arch}) …")
    model = build_model(device, arch)
    ckpt  = os.path.join(cache_dir, f"{arch}_cifar10.pth")
    model = fine_tune_on_cifar10(
        model, os.path.join(cache_dir, "cifar10"),
        device, epochs=epochs, checkpoint_path=ckpt,
    )

    # ── 3. Logits ─────────────────────────────────────────────────────────────
    print(f"\n[3/6] Extracting logits on CIFAR-10 test set …")
    logits_cache = os.path.join(cache_dir, f"logits_test_{arch}.npy")
    logits_all   = extract_logits(model, testset, device, cache_path=logits_cache)
    probs_all    = torch.softmax(torch.tensor(logits_all), dim=1).numpy()

    acc = float((logits_all.argmax(1) == hard_labels).mean())
    print(f"  Test accuracy: {acc*100:.2f}%")

    # ── 4. Cal / test split (50/50, stratified by class) ─────────────────────
    print("\n[4/6] Splitting into cal / test (5000 / 5000) …")
    idx = np.arange(10000)
    cal_mask = np.zeros(10000, dtype=bool)
    for c in range(N_CLASSES):
        c_idx  = idx[hard_labels == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True

    idx_cal = idx[cal_mask]
    idx_te  = idx[~cal_mask]

    logits_cal = torch.tensor(logits_all[idx_cal], dtype=torch.float32)
    logits_te  = torch.tensor(logits_all[idx_te],  dtype=torch.float32)
    probs_cal  = probs_all[idx_cal]
    probs_te   = probs_all[idx_te]

    yh_cal = hard_labels[idx_cal]
    ys_cal = cifar10h[idx_cal]            # soft labels
    yh_te  = hard_labels[idx_te]
    ys_te  = cifar10h[idx_te]

    yh_cal_t = torch.tensor(yh_cal, dtype=torch.long)
    ys_cal_t = torch.tensor(ys_cal, dtype=torch.float32)

    print(f"  Cal n={len(idx_cal)}, Test n={len(idx_te)}")
    print(f"  Mean annotation entropy (test): {annotation_entropy(ys_te).mean():.4f}")

    # ── 5. Calibration ────────────────────────────────────────────────────────
    print("\n[5/6] Fitting calibration methods …")

    # Parametric — baselines (hard labels)
    ts = TemperatureScaling().fit(logits_cal, yh_cal_t)
    ps = PlattScaling(N_CLASSES).fit(logits_cal, yh_cal_t)

    print(f"  T (TS):   {ts.T:.4f}  {'← < 1 (wrong direction!)' if ts.T < 1 else ''}")

    # Parametric — ours (soft labels)
    slts = SoftLabelTS().fit(logits_cal, ys_cal_t)
    mcts = MonteCarloTS(n_samples=50).fit(logits_cal, ys_cal_t)
    vs   = VectorScaling(N_CLASSES).fit(logits_cal, ys_cal_t)

    print(f"  T (SLTS): {slts.T:.4f}")
    print(f"  T (MCTS): {mcts.T:.4f}")

    # Non-parametric — baseline (hard labels)
    hb_hard = HardHistogramBinning(n_bins=n_bins).fit(probs_cal, yh_cal)

    # Non-parametric — ours (soft labels)
    hb = SoftHistogramBinning(n_bins=n_bins).fit(probs_cal, ys_cal)
    ir = SoftIsotonicRegression().fit(probs_cal, ys_cal)

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    print("\n[6/6] Evaluating …")

    # Parametric: get full probability vectors
    p_ts   = apply_parametric(ts,   logits_all[idx_te])
    p_ps   = apply_parametric(ps,   logits_all[idx_te])
    p_slts = apply_parametric(slts, logits_all[idx_te])
    p_mcts = apply_parametric(mcts, logits_all[idx_te])
    p_vs   = apply_parametric(vs,   logits_all[idx_te])

    main_results = []
    for name, p in [
        ("Uncalibrated",   probs_te),
        ("TS",             p_ts),
        ("Platt (PS)",     p_ps),
        ("MCTS (ours)",    p_mcts),
        ("SLTS (ours)",    p_slts),
        ("VS (ours)",      p_vs),
    ]:
        r = compute_all_metrics(p, yh_te, ys_te, n_bins=n_bins, name=name)
        r["temperature"] = (
            ts.T   if name == "TS"          else
            mcts.T if "MCTS" in name        else
            slts.T if "SLTS" in name        else None
        )
        main_results.append(r)

    # Non-parametric: confidence-based ECE only
    for name, method in [
        ("HB-Hard",       hb_hard),
        ("HB-Soft (ours)", hb),
        ("IR-Soft (ours)", ir),
    ]:
        cal_conf, pred = method.calibrate(probs_te)
        # Reconstruct a pseudo-prob matrix (argmax = pred, max = cal_conf)
        pseudo_probs = np.zeros_like(probs_te)
        for i in range(len(probs_te)):
            pseudo_probs[i, pred[i]] = cal_conf[i]
            # fill remaining with uniform
            rest = np.ones(N_CLASSES) * (1 - cal_conf[i]) / (N_CLASSES - 1)
            rest[pred[i]] = 0
            pseudo_probs[i] += rest

        r = compute_all_metrics(pseudo_probs, yh_te, ys_te, n_bins=n_bins, name=name)
        r["temperature"] = None
        main_results.append(r)

    print_results_table(main_results)

    # Stratified by ambiguity
    print("\n--- Stratified ECE-Soft (ambiguous vs. clear) ---")
    strat_results = {}
    for name, p in [
        ("TS",          p_ts),
        ("Platt (PS)",  p_ps),
        ("SLTS (ours)", p_slts),
        ("MCTS (ours)", p_mcts),
    ]:
        s = ambiguity_split_ece(p, yh_te, ys_te, n_bins=n_bins)
        strat_results[name] = s
        print(f"  {name:<18}: amb={s.get('ece_soft_ambiguous', float('nan'))*100:.2f}%  "
              f"clear={s.get('ece_soft_clear', float('nan'))*100:.2f}%  "
              f"thresh={s['entropy_threshold']:.3f}")

    # Quantile stratification
    quant_results = {}
    for name, p in [
        ("TS", p_ts), ("Platt (PS)", p_ps),
        ("SLTS (ours)", p_slts), ("MCTS (ours)", p_mcts),
    ]:
        quant_results[name] = stratified_ece_soft(p, ys_te, n_bins=n_bins, n_quantiles=4)

    # Save all results
    output = {
        "main_results":   main_results,
        "strat_results":  strat_results,
        "quant_results":  quant_results,
        "ts_temperature":   ts.T,
        "slts_temperature": slts.T,
        "mcts_temperature": mcts.T,
        "test_accuracy":  acc,
        "n_cal":          int(len(idx_cal)),
        "n_test":         int(len(idx_te)),
        "n_bins":         n_bins,
        "seed":           seed,
    }
    out_path = Path(results_dir) / f"cifar10h_results_{arch}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10H calibration experiment")
    p.add_argument("--cache-dir",   default="./cache",   help="directory for downloaded data and logit cache")
    p.add_argument("--results-dir", default="./results", help="directory for output JSON / figures")
    p.add_argument("--n-bins",      type=int, default=15, help="ECE bins (default 15)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs",      type=int, default=30,
                   help="fine-tuning epochs (ignored if checkpoint already exists)")
    p.add_argument("--arch",        default="resnet50",
                   choices=["resnet50", "vit_b16"],
                   help="backbone architecture (default: resnet50)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {args.device}")
    run_experiment(args)
