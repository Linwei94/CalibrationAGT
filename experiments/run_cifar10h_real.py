#!/usr/bin/env python
"""
REAL, fully-reproducible CIFAR-10H calibration under ambiguous ground truth
(TPAMI resubmission — reproducibility + real-human-disagreement validation).

Uses ONLY public Hugging Face assets (no proprietary checkpoints), so a reviewer can
re-run it end-to-end:
  - Model: aaraki/vit-base-patch16-224-in21k-finetuned-cifar10 (ViT-B/16 fine-tuned on
    CIFAR-10; its id2label is the canonical order [airplane..truck]).
  - Data:  MKZuziak/cifar10h parquet (10,000 rows = CIFAR-10 test set) with the real
    per-image human vote histograms (`expert_counts`), already in canonical class order.

Pipeline: ViT -> (N,10) logits (canonical order) ; soft label = expert_counts/sum ;
voted label = argmax(counts) ; correctness gate (top-1 vs voted > 0.90) ; stratified
50/50 split (seed 42) ; save a depth-analysis bundle ; fit TS/SLTS/MCTS S=1/LS-TS/
Dirichlet-Soft/IR-Soft and report ECE_true/Brier/NLL on the test split.

Then: python analyze_depth.py --bundle <out>/cifar10h_vit_real.npz   for the D1-D4 diagnostics.

Usage:  python run_cifar10h_real.py [--out results/bundles] [--device cuda]
"""

import argparse, io, json, os, sys
from pathlib import Path
import numpy as np
import torch

# The aaraki checkpoint ships only pytorch_model.bin; HF's Xet backend can stall on it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_all_metrics, annotation_entropy, print_results_table
from calibration import (TemperatureScaling, SoftLabelTS, MonteCarloTS, LabelSmoothTS,
                         DirichletCalibration, SoftIsotonicRegression, apply_parametric)
from analyze_depth import save_bundle

MODEL_ID = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
N_CLASSES = 10
CANONICAL = ["airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bundles", help="dir for the .npz bundle")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import ViTForImageClassification, AutoImageProcessor
    from PIL import Image
    import pandas as pd

    print(f"[1] model {MODEL_ID}")
    model = ViTForImageClassification.from_pretrained(MODEL_ID).to(args.device).eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    order = [model.config.id2label[i] for i in range(N_CLASSES)]
    assert order == CANONICAL, f"model label order {order} != canonical CIFAR-10"

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

    print(f"[3] ViT forward on {args.device}")
    logits = np.zeros((N, N_CLASSES), np.float32)
    with torch.no_grad():
        for s in range(0, N, args.batch):
            e = min(s + args.batch, N)
            pv = processor(images=images[s:e], return_tensors="pt")["pixel_values"].to(args.device)
            logits[s:e] = model(pixel_values=pv).logits.float().cpu().numpy()

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
    bundle = str(Path(args.out) / "cifar10h_vit_real.npz")
    save_bundle(bundle, name="cifar10h_vit_real", n_classes=N_CLASSES,
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
