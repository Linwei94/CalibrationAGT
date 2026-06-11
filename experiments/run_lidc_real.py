#!/usr/bin/env python
"""
REAL multi-reader MEDICAL calibration on LIDC-IDRI.

Recipe: /tmp/real4/acq_lidc.md
Dataset: ykeselman/lidc-idri-patches  (data/train-00000-of-00001.parquet, 40187 rows;
         one row per (annotation, z-slice); per-reader malignancy IS available -> feasible).

Pipeline
--------
[1] Load parquet. A physical NODULE = group by (patient_id, scan_id, cluster_id).
    Each distinct annotation_id within a nodule = one radiologist; its `malignancy`
    is constant across that annotation's slices (verified: 0 / 6859 annotations have
    >1 value). Keep nodules with >=3 readers (1392). Representative crop per reader =
    the annotation's median-z slice; nodule image = first reader's representative crop.
    K=2 binary: malignant=1 if rating_idx>=3 (0-based LIDC, i.e. LIDC>=4) else 0.
    soft label = normalized histogram of per-reader binaries [P(benign), P(malig)];
    voted = majority (tie -> malignant, i.e. argmax with malignant winning ties).

[2] resnet18 (torchvision IMAGENET1K_V1 pretrained), 224 input (64x64 crops upsampled),
    class-weighted CE, ~20 epochs, cuda. Patient-level 60/20/20 split (seed 42) so no
    patient leaks across train/cal/test. Extract cal+test logits.

[3] CORRECTNESS GATE: test top-1 (argmax logits vs voted) > 0.65. Else report + stop.

[4] save_bundle -> /tmp/real4/bundles/lidc_real.npz  (n_classes=2).

[5] Fit TS(voted)/SLTS/MCTS S=1/LS-TS/Dirichlet-Soft/IR-Soft on cal; report
    ECE_true/Brier/NLL on test via metrics.compute_all_metrics; print fitted T's.
"""

import argparse, io, json, os, sys
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

EXP_DIR = "/root/CalibrationAGT/experiments"
sys.path.insert(0, EXP_DIR)
from metrics import compute_all_metrics, annotation_entropy, print_results_table
from calibration import (TemperatureScaling, SoftLabelTS, MonteCarloTS, LabelSmoothTS,
                         DirichletCalibration, SoftIsotonicRegression, apply_parametric)
from analyze_depth import save_bundle

DATASET_ID = "ykeselman/lidc-idri-patches"
PARQUET = "data/train-00000-of-00001.parquet"
N_CLASSES = 2          # K=2 binary malignancy
MIN_READERS = 3
INPUT_SIZE = 224       # 64x64 crops -> upsample to 224 for ImageNet-pretrained resnet18
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def rating_to_class(rating_idx: int) -> int:
    """K=2 binary per reader: malignant=1 if rating_idx>=3 (0-based LIDC>=4) else 0."""
    return 1 if int(rating_idx) >= 3 else 0


def load_nodules(verbose=True):
    """Return per-nodule: PIL image, soft label (K,), voted (int), patient_id, n_readers."""
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import pandas as pd

    pq = hf_hub_download(DATASET_ID, PARQUET, repo_type="dataset")
    df = pd.read_parquet(pq)
    if verbose:
        print(f"[1] {DATASET_ID}: {len(df)} (annotation,slice) rows, "
              f"{df['annotation_id'].nunique()} annotations, "
              f"{df.groupby(['patient_id','scan_id','cluster_id']).ngroups} nodules")
    # Safety re-verify: malignancy constant within annotation_id.
    bad = int((df.groupby("annotation_id")["malignancy"].nunique() > 1).sum())
    assert bad == 0, f"{bad} annotations have non-constant malignancy -> recipe assumption broken"

    images, soft, voted, pids, nread, ents = [], [], [], [], [], []
    for (pid, sid, cid), nod in df.groupby(["patient_id", "scan_id", "cluster_id"]):
        ann_ids = sorted(nod["annotation_id"].unique())
        if len(ann_ids) < MIN_READERS:
            continue
        # per-reader binary class (rating constant within annotation -> take first)
        per_reader = []
        rep_crop = None
        for j, aid in enumerate(ann_ids):
            sub = nod[nod["annotation_id"] == aid]
            rating = int(sub["malignancy"].iloc[0])
            per_reader.append(rating_to_class(rating))
            if j == 0:
                # representative crop = this annotation's median-z slice
                sub_sorted = sub.sort_values("z")
                mid = sub_sorted.iloc[len(sub_sorted) // 2]
                rep_crop = Image.open(io.BytesIO(mid["image_8bit"]["bytes"])).convert("RGB")
        per_reader = np.asarray(per_reader, dtype=np.int64)
        hist = np.bincount(per_reader, minlength=N_CLASSES).astype(np.float64)
        s = hist / hist.sum()
        # voted = majority; tie -> malignant (argmax breaks toward higher index here).
        v = 1 if s[1] >= s[0] else 0
        images.append(rep_crop)
        soft.append(s)
        voted.append(v)
        pids.append(int(pid))
        nread.append(len(ann_ids))
    soft = np.asarray(soft, np.float64)
    voted = np.asarray(voted, np.int64)
    pids = np.asarray(pids, np.int64)
    nread = np.asarray(nread, np.int64)
    if verbose:
        H = annotation_entropy(soft)
        print(f"    kept {len(images)} nodules with >={MIN_READERS} readers; "
              f"mean readers/nodule={nread.mean():.3f}; "
              f"voted balance benign={int((voted==0).sum())} malignant={int((voted==1).sum())}; "
              f"mean annotation entropy={H.mean():.4f} nats")
    return images, soft, voted, pids, nread


def patient_split(pids, seed=42, fracs=(0.6, 0.2, 0.2)):
    """Patient-level 60/20/20 split: every nodule of a patient goes to one fold."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(pids)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(fracs[0] * n))
    n_ca = int(round(fracs[1] * n))
    tr_p = set(uniq[:n_tr].tolist())
    ca_p = set(uniq[n_tr:n_tr + n_ca].tolist())
    te_p = set(uniq[n_tr + n_ca:].tolist())
    tr = np.array([i for i, p in enumerate(pids) if p in tr_p], dtype=np.int64)
    ca = np.array([i for i, p in enumerate(pids) if p in ca_p], dtype=np.int64)
    te = np.array([i for i, p in enumerate(pids) if p in te_p], dtype=np.int64)
    return tr, ca, te


class NoduleDS(torch.utils.data.Dataset):
    def __init__(self, images, labels, tf):
        self.images, self.labels, self.tf = images, labels, tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.tf(self.images[i]), int(self.labels[i])


def build_resnet18(device):
    import torchvision
    from torchvision.models import ResNet18_Weights
    m = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(m.fc.in_features, N_CLASSES)
    return m.to(device)


def train(model, images, voted, tr, ca, device, epochs, batch, seed, lr=1e-4):
    import torchvision.transforms as T
    torch.manual_seed(seed); np.random.seed(seed)
    tf_train = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    tf_eval = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    tr_imgs = [images[i] for i in tr]
    tr_lab = voted[tr]
    ds = NoduleDS(tr_imgs, tr_lab, tf_train)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, drop_last=False)

    # class-weighted CE (inverse frequency on the TRAIN voted labels)
    counts = np.bincount(tr_lab, minlength=N_CLASSES).astype(np.float64)
    w = counts.sum() / (N_CLASSES * np.clip(counts, 1, None))
    weight = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[2] train resnet18 (224, ImageNet pretrained); n_train={len(tr)} "
          f"class counts={counts.astype(int).tolist()} CE weight={np.round(w,3).tolist()}")
    crit = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        model.train()
        tot, correct, lsum = 0, 0, 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            lsum += loss.item() * len(y)
            correct += int((out.argmax(1) == y).sum()); tot += len(y)
        sched.step()
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep+1:2d}/{epochs}  loss={lsum/tot:.4f}  train_acc={correct/tot:.4f}")
    return tf_eval


@torch.no_grad()
def extract_logits(model, images, idx, tf_eval, device, batch):
    model.eval()
    out = np.zeros((len(idx), N_CLASSES), np.float32)
    for s in range(0, len(idx), batch):
        e = min(s + batch, len(idx))
        x = torch.stack([tf_eval(images[idx[i]]) for i in range(s, e)]).to(device)
        out[s:e] = model(x).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/real4/bundles")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    images, soft, voted, pids, nread = load_nodules()

    tr, ca, te = patient_split(pids, seed=args.seed)
    # no patient leakage assertion
    assert set(pids[tr]).isdisjoint(set(pids[ca])) and set(pids[tr]).isdisjoint(set(pids[te])) \
        and set(pids[ca]).isdisjoint(set(pids[te])), "patient leakage across folds!"
    print(f"    patient-level split (seed {args.seed}): "
          f"train={len(tr)} cal={len(ca)} test={len(te)} nodules; "
          f"patients train={len(set(pids[tr]))} cal={len(set(pids[ca]))} test={len(set(pids[te]))}")

    model = build_resnet18(args.device)
    tf_eval = train(model, images, voted, tr, ca, args.device, args.epochs, args.batch, args.seed)

    logits_cal = extract_logits(model, images, ca, tf_eval, args.device, args.batch)
    logits_te = extract_logits(model, images, te, tf_eval, args.device, args.batch)

    top1 = float((logits_te.argmax(1) == voted[te]).mean())
    print(f"[3] CORRECTNESS GATE: test top-1 vs voted = {top1:.4f}  (need > 0.65)")
    if not (top1 > 0.65):
        print(f"GATE FAILED: test top-1 {top1:.4f} <= 0.65 -> binary malignancy not learnable here; STOP.")
        print(json.dumps({"status": "gate_failed", "dataset": DATASET_ID,
                          "n_nodules": len(images), "test_top1": top1}))
        sys.exit(2)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    bundle = str(Path(args.out) / "lidc_real.npz")
    save_bundle(bundle, name="lidc_real", n_classes=N_CLASSES,
                logits_cal=logits_cal, logits_te=logits_te,
                soft_cal=soft[ca], soft_te=soft[te],
                hard_cal=voted[ca], hard_te=voted[te])
    print(f"[4] bundle -> {bundle}")

    # [5] fit calibrators on cal, eval on test
    lc = torch.tensor(logits_cal, dtype=torch.float32)
    yh = torch.tensor(voted[ca], dtype=torch.long)
    ys = torch.tensor(soft[ca], dtype=torch.float32)
    lt = logits_te
    probs = {"Uncalibrated": torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()}
    ts = TemperatureScaling().fit(lc, yh);   probs["TS"] = apply_parametric(ts, lt)
    slts = SoftLabelTS().fit(lc, ys);        probs["SLTS"] = apply_parametric(slts, lt)
    mc = MonteCarloTS(n_samples=1, seed=args.seed).fit(lc, ys); probs["MCTS S=1"] = apply_parametric(mc, lt)
    ls = LabelSmoothTS().fit(lc, yh);        probs["LS-TS"] = apply_parametric(ls, lt)
    dc = DirichletCalibration(N_CLASSES).fit_soft(lc, ys); probs["Dirichlet-Soft"] = apply_parametric(dc, lt)
    pc = torch.softmax(lc, 1).numpy()
    ir = SoftIsotonicRegression().fit(pc, soft[ca])
    pt = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()
    conf, pred = ir.calibrate(pt); pir = np.zeros_like(pt)
    for i in range(len(pt)):
        pir[i, pred[i]] = conf[i]
        rest = np.ones(N_CLASSES) * (1 - conf[i]) / (N_CLASSES - 1); rest[pred[i]] = 0
        pir[i] += rest
    probs["IR-Soft"] = pir

    temps = {"TS": ts.T, "SLTS": slts.T, "MCTS S=1": mc.T, "LS-TS": ls.T}
    print(f"[5] fitted T's: TS={ts.T:.4f} SLTS={slts.T:.4f} MCTS S=1={mc.T:.4f} LS-TS={ls.T:.4f}")
    results = [compute_all_metrics(p, voted[te], soft[te], name=n) for n, p in probs.items()]
    print_results_table(results)

    # machine-readable summary (ECE_true == ece_sampled; Brier/NLL true-label == *_sampled)
    summary = {
        "status": "ok", "dataset": DATASET_ID, "n_nodules_kept": len(images),
        "min_readers": MIN_READERS, "mean_readers_per_nodule": float(nread.mean()),
        "n_train": len(tr), "n_cal": len(ca), "n_test": len(te),
        "test_top1_vs_voted": top1,
        "mean_annotation_entropy_nats": float(annotation_entropy(soft).mean()),
        "voted_balance": {"benign": int((voted == 0).sum()), "malignant": int((voted == 1).sum())},
        "temperatures": temps, "bundle_path": bundle,
        "metrics": [{"method": r["name"],
                     "ECE_true": r["ece_sampled"], "ECE_soft": r["ece_soft"], "ECE_hard": r["ece_hard"],
                     "Brier_true": r["brier_sampled"], "Brier_soft": r["brier_soft"],
                     "NLL_true": r["nll_sampled"], "NLL_soft": r["nll_soft"]} for r in results],
    }
    print("JSON_SUMMARY_BEGIN")
    print(json.dumps(summary, indent=2))
    print("JSON_SUMMARY_END")


if __name__ == "__main__":
    main()
