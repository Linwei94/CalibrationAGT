#!/usr/bin/env python
"""
REAL ImageNet-scale calibration-under-ambiguous-ground-truth (AE #2: "does it scale?").

Data sources (all ungated, HF-reachable):
  * IMAGES + original ILSVRC labels:
        zera09/imagenet_val_full  (HF parquet mirror of the ILSVRC-2012 val set).
        Each row carries image.path = "ILSVRC2012_val_XXXXXXXX_<wnid>.JPEG", so the
        original ILSVRC val index AND the synset are recoverable per image. The row's
        `label` field is the standard timm/torchvision class index (synset-sorted
        0..999); we verified label == wnid->index on every probed row (3572/3572).
  * ReaL / multi-label reassessed labels:
        timm.data._info/imagenet_real_labels.json  (Ross Wightman's verbatim copy of
        google-research/reassessed-imagenet `real.json`; raw.githubusercontent is
        proxy-blocked here, the timm copy is identical). A length-50000 list, entry i =
        SET of valid 0-indexed classes for ILSVRC2012_val_{i+1:08d}.JPEG. 46837/50000
        entries non-empty; 3163 empty (dropped). Multi-label sizes 1..>7.

Alignment (verified on shard 0, 3318 non-empty): the original ILSVRC class sits in
  ReaL[val_index] for 90.8% of images -- exactly the ~9% reassessment rate from the
  "Are we done with ImageNet?" paper -- confirming filename-index == ReaL-index and a
  shared class indexing. Alignment is therefore by ILSVRC filename index, exact.

Model: timm resnet50 (pretrained, 1000-way logits in standard ILSVRC order), with its
  own resolve_data_config / create_transform eval pipeline.

Soft labels: for each image with a NON-EMPTY ReaL set -> uniform over that set
  (normalized indicator over 1000 classes). voted/hard label = original ILSVRC top-1.

Calibrators (scalable only; Dirichlet/VS skipped -- K=1000 infeasible):
  TS (voted), SLTS, MCTS S=1, LS-TS, IR-Soft.
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

EXP_DIR = "/root/CalibrationAGT/experiments"
sys.path.insert(0, EXP_DIR)
from metrics import compute_all_metrics, annotation_entropy, print_results_table
from calibration import (TemperatureScaling, SoftLabelTS, MonteCarloTS, LabelSmoothTS,
                         SoftIsotonicRegression, apply_parametric)
from analyze_depth import save_bundle

IMG_DS = "zera09/imagenet_val_full"
REAL_JSON = "/miniconda/envs/llarp/lib/python3.9/site-packages/timm/data/_info/imagenet_real_labels.json"
N_CLASSES = 1000
FNAME_RE = re.compile(r"ILSVRC2012_val_(\d{8})_(n\d+)\.JPEG")


def dl(ds, fn, repo_type="dataset", tries=5):
    from huggingface_hub import hf_hub_download
    last = None
    for a in range(tries):
        try:
            return hf_hub_download(ds, fn, repo_type=repo_type, force_download=(a > 0))
        except Exception as e:
            last = e; print(f"    retry {fn} ({a}): {type(e).__name__}"); time.sleep(3)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/real3/bundles")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0, help="0 = use all val images")
    args = ap.parse_args()

    import timm
    from timm.data import resolve_data_config, create_transform
    from huggingface_hub import HfApi
    import pyarrow.parquet as pq
    from PIL import Image
    import io, pkgutil

    # ---- ReaL labels --------------------------------------------------------
    print("[1] ReaL labels (timm imagenet_real_labels.json)")
    real = json.load(open(REAL_JSON))
    assert len(real) == 50000, f"expected 50000 ReaL entries, got {len(real)}"
    wnids = pkgutil.get_data("timm.data", "_info/imagenet_synsets.txt").decode().split()
    wnid2idx = {w: i for i, w in enumerate(wnids)}
    assert len(wnids) == 1000

    # ---- model + transform --------------------------------------------------
    print(f"[3] model timm {args.model} (pretrained)")
    model = timm.create_model(args.model, pretrained=True).to(args.device).eval()
    cfg = resolve_data_config({}, model=model)
    tf = create_transform(**cfg)
    print(f"    eval transform cfg: input_size={cfg.get('input_size')} "
          f"crop_pct={cfg.get('crop_pct')}")

    # ---- images -------------------------------------------------------------
    print(f"[2] images {IMG_DS}")
    api = HfApi()
    pqs = sorted(s.rfilename for s in api.dataset_info(IMG_DS).siblings
                 if s.rfilename.endswith(".parquet"))
    print(f"    {len(pqs)} parquet shards")

    val_idx, hard, soft_sets, imgs_logits = [], [], [], []
    label_synset_ok = 0
    N_seen = 0
    logits_all = []
    keep_validx = []
    keep_hard = []
    keep_real = []

    for si, fn in enumerate(pqs):
        p = dl(IMG_DS, fn)
        t = pq.read_table(p, columns=["image", "label"]).to_pydict()
        paths = [d["path"] for d in t["image"]]
        byts = [d["bytes"] for d in t["image"]]
        labs = t["label"]
        # batch-forward this shard
        batch_imgs, batch_meta = [], []

        def flush():
            if not batch_imgs:
                return
            x = torch.stack(batch_imgs).to(args.device)
            with torch.no_grad():
                lo = model(x).float().cpu().numpy()
            for j, (vi, hl, rs) in enumerate(batch_meta):
                logits_all.append(lo[j]); keep_validx.append(vi)
                keep_hard.append(hl); keep_real.append(rs)
            batch_imgs.clear(); batch_meta.clear()

        for pth, bb, lab in zip(paths, byts, labs):
            m = FNAME_RE.match(pth)
            if m is None:
                continue
            vi = int(m.group(1)) - 1
            cls = wnid2idx[m.group(2)]
            label_synset_ok += int(lab == cls)
            N_seen += 1
            rs = real[vi]
            if not rs:          # empty ReaL set -> drop
                continue
            if args.max_images and len(logits_all) + len(batch_meta) >= args.max_images:
                continue
            im = Image.open(io.BytesIO(bb)).convert("RGB")
            batch_imgs.append(tf(im))
            batch_meta.append((vi, cls, rs))
            if len(batch_imgs) >= args.batch:
                flush()
        flush()
        print(f"    shard {si+1}/{len(pqs)} done; kept so far {len(logits_all)}")
        if args.max_images and len(logits_all) >= args.max_images:
            break

    logits = np.asarray(logits_all, dtype=np.float32)
    hard = np.asarray(keep_hard, dtype=np.int64)
    N = len(hard)
    print(f"    seen {N_seen} images; label==synset class on {label_synset_ok}/{N_seen}")
    assert label_synset_ok == N_seen, "label field disagrees with synset -> class-order bug"
    print(f"    kept {N} images with non-empty ReaL sets")

    # ---- soft labels (uniform over ReaL set) --------------------------------
    soft = np.zeros((N, N_CLASSES), dtype=np.float32)
    for i, rs in enumerate(keep_real):
        soft[i, rs] = 1.0 / len(rs)
    mean_ent = float(annotation_entropy(soft).mean())
    print(f"[5] soft labels built; mean annotation entropy {mean_ent:.4f} nats")

    # ---- correctness gate ---------------------------------------------------
    top1 = float((logits.argmax(1) == hard).mean())
    print(f"[4] correctness gate: top-1 vs original ILSVRC label = {top1:.4f}")
    assert top1 > 0.70, f"top-1 {top1:.4f} <= 0.70 -> alignment/preprocessing wrong"

    # ---- stratified-ish 50/50 split -----------------------------------------
    print(f"[6] stratified 50/50 split (seed {args.seed})")
    rng = np.random.default_rng(args.seed)
    cal, te = [], []
    for c in range(N_CLASSES):
        ic = np.where(hard == c)[0]
        rng.shuffle(ic)
        h = len(ic) // 2
        cal.append(ic[:h]); te.append(ic[h:])
    cal = np.sort(np.concatenate(cal)); te = np.sort(np.concatenate(te))
    print(f"    cal {len(cal)}  test {len(te)}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    bundle = str(Path(args.out) / "imagenet_real_resnet50.npz")
    save_bundle(bundle, name="imagenet_real_resnet50", n_classes=N_CLASSES,
                logits_cal=logits[cal], logits_te=logits[te],
                soft_cal=soft[cal], soft_te=soft[te],
                hard_cal=hard[cal], hard_te=hard[te])

    # ---- fit + evaluate scalable calibrators --------------------------------
    print("[7] fit + evaluate (scalable calibrators only)")
    lc = torch.tensor(logits[cal], dtype=torch.float32)
    yh = torch.tensor(hard[cal], dtype=torch.long)
    ys = torch.tensor(soft[cal], dtype=torch.float32)
    lt = logits[te]

    probs = {"Uncalibrated": torch.softmax(torch.tensor(lt), 1).numpy()}
    ts = TemperatureScaling().fit(lc, yh);            probs["TS (voted)"] = apply_parametric(ts, lt)
    slts = SoftLabelTS().fit(lc, ys);                 probs["SLTS"] = apply_parametric(slts, lt)
    mc = MonteCarloTS(n_samples=1, seed=args.seed).fit(lc, ys); probs["MCTS S=1"] = apply_parametric(mc, lt)
    ls = LabelSmoothTS().fit(lc, yh);                 probs["LS-TS"] = apply_parametric(ls, lt)

    pc = torch.softmax(lc, 1).numpy()
    ir = SoftIsotonicRegression().fit(pc, soft[cal])
    pt = torch.softmax(torch.tensor(lt), 1).numpy()
    conf, pred = ir.calibrate(pt)
    pir = np.zeros_like(pt)
    for i in range(len(pt)):
        c = float(conf[i]); pir[i, pred[i]] = c
        rest = np.ones(N_CLASSES) * (1 - c) / (N_CLASSES - 1)
        rest[pred[i]] = 0.0
        pir[i] += rest
    probs["IR-Soft"] = pir

    print(f"    T: TS={ts.T:.4f} SLTS={slts.T:.4f} MCTS={mc.T:.4f} LS-TS={ls.T:.4f}")
    results = [compute_all_metrics(p, hard[te], soft[te], name=n) for n, p in probs.items()]
    print_results_table(results)

    # persist a small summary for the final report
    summary = {
        "image_dataset": IMG_DS, "real_labels": "timm imagenet_real_labels.json",
        "model": f"timm/{args.model}", "N_kept": N, "N_seen": N_seen,
        "top1_acc": top1, "mean_annotation_entropy": mean_ent,
        "T": {"TS": ts.T, "SLTS": slts.T, "MCTS_S1": mc.T, "LS-TS": ls.T},
        "bundle": bundle,
        "metrics": {r["name"]: {k: r[k] for k in
                    ("ece_soft", "adaptive_ece_true", "brier_soft", "nll_soft")}
                    for r in results},
    }
    json.dump(summary, open("/tmp/real3/summary.json", "w"), indent=2)
    print("\n[done] summary -> /tmp/real3/summary.json")


if __name__ == "__main__":
    main()
