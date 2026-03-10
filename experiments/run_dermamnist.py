"""
DermaMNIST skin disease calibration experiment.

Analogous to the skin-condition case study in Stutz et al. (2023)
"Conformal Prediction under Ambiguous Ground Truth", but for calibration.

Dataset
-------
DermaMNIST (MedMNIST, derived from HAM10000; Tschandl et al. 2018).
  Train: 7,007 images  |  Val: 1,003  |  Test: 2,005
  7 classes (AK, BCC, BKL, DF, Mel, NV, Vasc)

Annotator model
---------------
Synthetic annotations generated from a clinically-calibrated 7×7
confusion matrix (overall inter-reader agreement ≈ 64.7%), consistent
with published dermatologist accuracy on this task (Haenssle et al. 2018;
Tschandl et al. 2019).  Each image receives K=5 synthetic annotator labels.

Usage
-----
    pip install medmnist
    python run_dermamnist.py [--device cuda] [--epochs 30]
    # Results saved to results/dermamnist_results.json
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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    TemperatureScaling, PlattScaling, DirichletCalibration,
    SoftPlattScaling,
    SoftLabelTS, MonteCarloTS, VectorScaling,
    PseudoSoftLabelTS, LabelSmoothTS,
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

N_CLASSES   = 7
CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "Mel", "NV", "Vasc"]
N_ANNOTATORS = 5   # synthetic annotator labels per image

# Clinically-inspired annotator confusion matrix.
# Row = consensus (hard) label; Col = individual annotator's label.
# Diagonal ≈ agreement rate per class; calibrated so that the overall
# weighted agreement ≈ 64.7%, matching published dermatologist performance
# on the HAM10000 7-class task (Haenssle et al. 2018; Tschandl et al. 2019).
#
#        AK     BCC    BKL    DF     Mel    NV     Vasc
CONFUSION = np.array([
    [0.62,  0.07,  0.16,  0.02,  0.07,  0.05,  0.01],  # AK  (actinic keratoses)
    [0.07,  0.73,  0.09,  0.03,  0.04,  0.03,  0.01],  # BCC (basal cell carcinoma)
    [0.12,  0.06,  0.62,  0.05,  0.08,  0.06,  0.01],  # BKL (benign keratosis-like)
    [0.02,  0.03,  0.04,  0.83,  0.03,  0.04,  0.01],  # DF  (dermatofibroma)
    [0.04,  0.04,  0.07,  0.02,  0.63,  0.19,  0.01],  # Mel (melanoma) ← most confusion
    [0.03,  0.02,  0.06,  0.04,  0.14,  0.70,  0.01],  # NV  (melanocytic nevi)
    [0.01,  0.01,  0.02,  0.02,  0.01,  0.01,  0.92],  # Vasc (vascular lesions)
], dtype=np.float64)
# Sanity check: each row sums to 1.0
assert np.allclose(CONFUSION.sum(axis=1), 1.0), "Confusion rows must sum to 1"

# ImageNet normalisation (used since we fine-tune from ImageNet pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────────────────────
# Annotator soft-label generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_soft_labels(
    hard_labels: np.ndarray,
    n_annotators: int = N_ANNOTATORS,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate n_annotators annotations per image using the confusion matrix.

    For image i with consensus label y_i, each annotator samples a label from
    Categorical(CONFUSION[y_i]).  The soft label is the empirical distribution
    over annotator responses.

    Returns
    -------
    soft : (N, K) float array of soft label distributions (rows sum to 1).
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
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(arch: str = "resnet18") -> nn.Module:
    """Build pretrained backbone with N_CLASSES head. arch: 'resnet18' | 'vit_s16'."""
    if arch == "vit_s16":
        import timm
        model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=N_CLASSES)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def get_loaders(data_root: str = "./data", batch_size: int = 64):
    """Load DermaMNIST splits via the medmnist library."""
    try:
        from medmnist import DermaMNIST
    except ImportError:
        raise ImportError(
            "medmnist is required. Install with: pip install medmnist"
        )

    tfm = T.Compose([
        T.ToTensor(),
        T.Resize(224, antialias=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    kw = dict(download=True, root=data_root, as_rgb=True)
    train_ds = DermaMNIST(split="train", transform=tfm, **kw)
    val_ds   = DermaMNIST(split="val",   transform=tfm, **kw)
    test_ds  = DermaMNIST(split="test",  transform=tfm, **kw)

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)
    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)


def _get_labels(loader: DataLoader) -> np.ndarray:
    """Collect integer class labels from a DataLoader."""
    all_labels = []
    for _, y in loader:
        if isinstance(y, (list, tuple)):
            y = y[0]
        all_labels.append(y.numpy().reshape(-1))
    return np.concatenate(all_labels).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 30,
    lr: float = 1e-4,
) -> nn.Module:
    """Fine-tune ResNet-18 with class-weighted cross-entropy (handles class imbalance)."""
    # Class weights: inverse frequency
    train_labels  = _get_labels(train_loader)
    counts        = np.bincount(train_labels, minlength=N_CLASSES).astype(float)
    weights       = counts.sum() / (N_CLASSES * counts)
    weights_t     = torch.tensor(weights, dtype=torch.float32, device=device)

    model   = model.to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit    = nn.CrossEntropyLoss(weight=weights_t)
    use_amp = str(device) == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

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
              f"val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  Best val accuracy: {best_acc*100:.2f}%")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Logit extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Extract raw logits (before softmax) for all examples in a loader."""
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
    p = argparse.ArgumentParser(description="DermaMNIST calibration experiment")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--data-root",   default="./data")
    p.add_argument("--cache-dir",   default="./cache")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--n-bins",      type=int, default=15)
    p.add_argument("--n-annotators",type=int, default=N_ANNOTATORS)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--skip-train",  action="store_true",
                   help="Load cached logits without training (fails if cache absent)")
    p.add_argument("--arch",        default="resnet18",
                   choices=["resnet18", "vit_s16"],
                   help="backbone architecture (default: resnet18)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    arch             = args.arch
    ckpt_path        = Path(args.cache_dir) / f"dermamnist_{arch}.pt"
    logits_val_path  = Path(args.cache_dir) / f"dermamnist_logits_val_{arch}.npy"
    logits_test_path = Path(args.cache_dir) / f"dermamnist_logits_test_{arch}.npy"

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("[1/5] Loading DermaMNIST …")
    train_loader, val_loader, test_loader = get_loaders(args.data_root, args.batch_size)

    val_labels  = _get_labels(val_loader)
    test_labels = _get_labels(test_loader)
    print(f"  Val n={len(val_labels)}, Test n={len(test_labels)}")
    print(f"  Class distribution (val):  {np.bincount(val_labels, minlength=N_CLASSES).tolist()}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    if logits_val_path.exists() and logits_test_path.exists() and args.skip_train:
        print("[2/5] Loading cached logits …")
        logits_val  = np.load(logits_val_path)
        logits_test = np.load(logits_test_path)
    else:
        print(f"[2/5] Fine-tuning {arch} ({args.epochs} epochs) …")
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
    print(f"  Val accuracy:  {val_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")

    # ── 3. Soft labels ────────────────────────────────────────────────────────
    print("[3/5] Generating synthetic soft labels …")
    ys_val  = generate_soft_labels(val_labels,  args.n_annotators, seed=args.seed)
    ys_test = generate_soft_labels(test_labels, args.n_annotators, seed=args.seed + 1)

    print(f"  Mean annotation entropy (val):  {annotation_entropy(ys_val).mean():.4f}")
    print(f"  Mean annotation entropy (test): {annotation_entropy(ys_test).mean():.4f}")

    # ── 4. Calibration ────────────────────────────────────────────────────────
    print("[4/5] Fitting calibration methods …")
    n_bins   = args.n_bins
    logits_v = torch.tensor(logits_val, dtype=torch.float32)
    yh_val_t = torch.tensor(val_labels, dtype=torch.long)
    ys_val_t = torch.tensor(ys_val,     dtype=torch.float32)

    # Parametric baselines (hard labels)
    ts   = TemperatureScaling().fit(logits_v, yh_val_t)
    ps   = PlattScaling(N_CLASSES).fit(logits_v, yh_val_t)
    dc_h = DirichletCalibration(N_CLASSES).fit_hard(logits_v, yh_val_t)

    # Parametric — ours (soft labels)
    slts = SoftLabelTS().fit(logits_v, ys_val_t)
    mcts = MonteCarloTS(n_samples=50).fit(logits_v, ys_val_t)
    vs   = VectorScaling(N_CLASSES).fit(logits_v, ys_val_t)
    dc_s = DirichletCalibration(N_CLASSES).fit_soft(logits_v, ys_val_t)
    sp_s = SoftPlattScaling(N_CLASSES).fit(logits_v, ys_val_t)

    # Annotation-free — ours (hard labels only; targets ECE_true)
    pslts = PseudoSoftLabelTS().fit(logits_v, yh_val_t)
    ls_ts = LabelSmoothTS().fit(logits_v, yh_val_t)

    print(f"  T(TS)={ts.T:.4f}  {'← T<1: wrong direction!' if ts.T < 1 else ''}")
    print(f"  T(SLTS)={slts.T:.4f}  T(MCTS)={mcts.T:.4f}")
    print(f"  T(PSLTS)={pslts.T:.4f}  T(LS-TS)={ls_ts.T:.4f}  (annotation-free)")

    # Non-parametric baselines (hard and soft)
    hb_hard = HardHistogramBinning(n_bins=n_bins).fit(probs_val, val_labels)
    hb_soft = SoftHistogramBinning(n_bins=n_bins).fit(probs_val, ys_val)
    ir_soft = SoftIsotonicRegression().fit(probs_val, ys_val)

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    print("[5/5] Evaluating …")
    # Parametric calibrators: get full probability vectors on test
    p_ts   = apply_parametric(ts,   logits_test)
    p_ps   = apply_parametric(ps,   logits_test)
    p_dc_h = apply_parametric(dc_h, logits_test)
    p_slts = apply_parametric(slts, logits_test)
    p_mcts = apply_parametric(mcts, logits_test)
    p_vs   = apply_parametric(vs,   logits_test)
    p_dc_s  = apply_parametric(dc_s,  logits_test)
    p_sp_s  = apply_parametric(sp_s,  logits_test)
    p_pslts = apply_parametric(pslts, logits_test)
    p_ls_ts = apply_parametric(ls_ts, logits_test)

    main_results = []
    for name, p in [
        ("Uncalibrated",          probs_test),
        ("TS",                    p_ts),
        ("Platt (PS)",            p_ps),
        ("Dirichlet-Hard",        p_dc_h),
        ("LabelSmooth-TS",        p_ls_ts),
        ("PSLTS (ours)",          p_pslts),
        ("MCTS (ours)",           p_mcts),
        ("SLTS (ours)",           p_slts),
        ("SoftPlatt (ours)",      p_sp_s),
        ("VS (ours)",             p_vs),
        ("Dirichlet-Soft (ours)", p_dc_s),
    ]:
        r = compute_all_metrics(p, test_labels, ys_test, n_bins=n_bins, name=name)
        r["temperature"] = (
            float(ts.T)    if name == "TS"               else
            float(mcts.T)  if "MCTS" in name             else
            float(pslts.T) if "PSLTS" in name            else
            float(slts.T)  if "SLTS" in name             else
            float(ls_ts.T) if "LabelSmooth" in name      else None
        )
        main_results.append(r)

    # Non-parametric: reconstruct pseudo-prob matrix (argmax=pred, max=cal_conf)
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

    # Stratified by ambiguity (annotation entropy quartile)
    quant_results = {}
    for name, p in [
        ("TS",          p_ts),
        ("Platt (PS)",  p_ps),
        ("SLTS (ours)", p_slts),
        ("MCTS (ours)", p_mcts),
    ]:
        quant_results[name] = stratified_ece_soft(p, ys_test, n_bins=n_bins, n_quantiles=4)

    # Ambiguous vs. clear split (median entropy threshold)
    print("\n--- Ambiguous vs. Clear split ---")
    for name, p in [("TS", p_ts), ("SLTS (ours)", p_slts)]:
        s = ambiguity_split_ece(p, test_labels, ys_test, n_bins=n_bins)
        print(f"  {name:<18}: amb={s.get('ece_soft_ambiguous', float('nan'))*100:.2f}%  "
              f"clear={s.get('ece_soft_clear', float('nan'))*100:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "dataset":          "DermaMNIST",
        "n_classes":        N_CLASSES,
        "class_names":      CLASS_NAMES,
        "n_annotators":     args.n_annotators,
        "main_results":     main_results,
        "quant_results":    quant_results,
        "ts_temperature":    float(ts.T),
        "slts_temperature":  float(slts.T),
        "mcts_temperature":  float(mcts.T),
        "pslts_temperature": float(pslts.T),
        "ls_ts_temperature": float(ls_ts.T),
        "val_accuracy":     float(val_acc),
        "test_accuracy":    float(test_acc),
        "n_val":            int(len(val_labels)),
        "n_test":           int(len(test_labels)),
        "n_bins":           n_bins,
        "seed":             args.seed,
    }
    out_path = Path(args.results_dir) / f"dermamnist_results_{arch}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Run `cd ../paper && python make_figures.py` to regenerate fig6_derm.pdf")


if __name__ == "__main__":
    main()
