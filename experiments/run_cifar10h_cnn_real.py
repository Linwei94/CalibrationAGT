#!/usr/bin/env python
"""
REAL CIFAR-10H calibration with an architecture-DIVERSE CNN checkpoint
(companion to run_cifar10h_real.py, which uses a ViT).

Model:  edadaltocg/resnet18_cifar10  (timm ResNet-18 with a CIFAR stem).
        The repo README's `timm.create_model("resnet18_cifar10", pretrained=True)`
        needs the `detectors` package (not installed). Instead we build the
        standard timm `resnet18`, patch the stem to CIFAR style (3x3 stride-1
        conv1, Identity maxpool), and load the published weights directly.
        load_state_dict(strict=True) -> 0 missing / 0 unexpected keys.

Preprocessing (per repo config.json):  32x32 RGB + CIFAR normalization
        mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010].
        NOT the ViT's 224 / ImageNet processor.

Class order: canonical CIFAR-10 [airplane..truck]. The recipe confirms the model
        already outputs index i == canonical class i, so the recipe permutation is
        the identity. We still apply PERM explicitly (logits = logits[:, PERM]) so
        the logits are guaranteed to be in canonical order regardless.

Data:  MKZuziak/cifar10h parquet (10,000 rows = CIFAR-10 test set) with the real
       per-image human vote histograms (`expert_counts`), in canonical class order.

Pipeline mirrors run_cifar10h_real.py exactly:
  CNN -> (N,10) canonical logits ; soft = expert_counts/sum ; voted = argmax(counts) ;
  correctness gate (top-1 vs voted > 0.90, assert) ; stratified 50/50 split (seed 42) ;
  save_bundle -> cifar10h_cnn_real.npz ; fit TS/SLTS/MCTS S=1/LS-TS/Dirichlet-Soft/IR-Soft
  ; print ECE_true/Brier/NLL table (metrics.compute_all_metrics).
"""

import argparse, io, os, sys
from pathlib import Path
import numpy as np
import torch

# The checkpoint ships only pytorch_model.bin; HF's Xet backend can stall on it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

EXP_DIR = "/root/CalibrationAGT/experiments"
sys.path.insert(0, EXP_DIR)
from metrics import compute_all_metrics, annotation_entropy, print_results_table
from calibration import (TemperatureScaling, SoftLabelTS, MonteCarloTS, LabelSmoothTS,
                         DirichletCalibration, SoftIsotonicRegression, apply_parametric)
from analyze_depth import save_bundle

MODEL_ID = "edadaltocg/resnet18_cifar10"
N_CLASSES = 10
CANONICAL = ["airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck"]
# Recipe class-order permutation: model index i == canonical class i (identity).
# Applied explicitly so logits[:, PERM] are in canonical CIFAR-10 order.
PERM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# CIFAR normalization from the repo config.json (NOT ImageNet / NOT std=[.247,.243,.261]).
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


def build_model(device):
    import timm
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(MODEL_ID, "pytorch_model.bin")
    sd = torch.load(ckpt, map_location="cpu")
    model = timm.create_model("resnet18", num_classes=N_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR stem
    model.maxpool = nn.Identity()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, \
        f"state_dict mismatch: missing={missing} unexpected={unexpected}"
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/real2/bundles", help="dir for the .npz bundle")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import torchvision.transforms as T
    from PIL import Image
    import pandas as pd

    print(f"[1] model {MODEL_ID} (timm resnet18, CIFAR stem)")
    model = build_model(args.device)
    tf = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])

    print("[2] data MKZuziak/cifar10h (real human votes)")
    pq = hf_hub_download("MKZuziak/cifar10h", "data/train-00000-of-00001.parquet",
                         repo_type="dataset")
    df = pd.read_parquet(pq)
    N = len(df)
    soft = np.zeros((N, N_CLASSES), np.float64)
    voted = np.zeros(N, np.int64)
    images = []
    for i in range(N):
        row = df.iloc[i]
        images.append(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))
        counts = np.asarray(row["expert_counts"], np.float64)
        soft[i] = counts / counts.sum()
        voted[i] = int(counts.argmax())
    print(f"    {N} rows; mean annotation entropy {annotation_entropy(soft).mean():.4f} nats")

    print(f"[3] CNN forward on {args.device}  (32x32 + CIFAR norm)")
    logits = np.zeros((N, N_CLASSES), np.float32)
    with torch.no_grad():
        for s in range(0, N, args.batch):
            e = min(s + args.batch, N)
            pv = torch.stack([tf(im) for im in images[s:e]]).to(args.device)
            logits[s:e] = model(pv).float().cpu().numpy()
    # canonical class-order permutation from the recipe (identity here)
    logits = logits[:, PERM]

    top1 = float((logits.argmax(1) == voted).mean())
    print(f"[4] correctness gate: top-1 vs voted = {top1:.4f}")
    assert top1 > 0.90, f"top-1 {top1:.4f} <= 0.90 -> preprocessing/label-order wrong"

    print(f"[5] stratified 50/50 split (seed {args.seed})")
    rng = np.random.default_rng(args.seed)
    cal, te = [], []
    for c in range(N_CLASSES):
        ic = np.where(voted == c)[0]; rng.shuffle(ic); h = len(ic) // 2
        cal.append(ic[:h]); te.append(ic[h:])
    cal = np.sort(np.concatenate(cal)); te = np.sort(np.concatenate(te))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    bundle = str(Path(args.out) / "cifar10h_cnn_real.npz")
    save_bundle(bundle, name="cifar10h_cnn_real", n_classes=N_CLASSES,
                logits_cal=logits[cal], logits_te=logits[te],
                soft_cal=soft[cal], soft_te=soft[te], hard_cal=voted[cal], hard_te=voted[te])

    print("[6] fit + evaluate")
    lc = torch.tensor(logits[cal], dtype=torch.float32)
    yh = torch.tensor(voted[cal], dtype=torch.long)
    ys = torch.tensor(soft[cal], dtype=torch.float32)
    lt = logits[te]
    probs = {"Uncalibrated": torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()}
    ts = TemperatureScaling().fit(lc, yh);   probs["TS"] = apply_parametric(ts, lt)
    slts = SoftLabelTS().fit(lc, ys);        probs["SLTS"] = apply_parametric(slts, lt)
    mc = MonteCarloTS(n_samples=1, seed=args.seed).fit(lc, ys); probs["MCTS S=1"] = apply_parametric(mc, lt)
    ls = LabelSmoothTS().fit(lc, yh);        probs["LS-TS"] = apply_parametric(ls, lt)
    dc = DirichletCalibration(N_CLASSES).fit_soft(lc, ys); probs["Dirichlet-Soft"] = apply_parametric(dc, lt)
    pc = torch.softmax(lc, 1).numpy()
    ir = SoftIsotonicRegression().fit(pc, soft[cal])
    pt = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()
    conf, pred = ir.calibrate(pt); pir = np.zeros_like(pt)
    for i in range(len(pt)):
        pir[i, pred[i]] = conf[i]; rest = np.ones(N_CLASSES) * (1 - conf[i]) / (N_CLASSES - 1)
        rest[pred[i]] = 0; pir[i] += rest
    probs["IR-Soft"] = pir
    print(f"    T: TS={ts.T:.3f} SLTS={slts.T:.3f} MCTS={mc.T:.3f} LS-TS={ls.T:.3f}")
    results = [compute_all_metrics(p, voted[te], soft[te], name=n) for n, p in probs.items()]
    print_results_table(results)


if __name__ == "__main__":
    main()
