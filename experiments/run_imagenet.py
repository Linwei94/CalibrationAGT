"""
Calibration under ambiguous ground truth at ImageNet scale (TPAMI resubmission).

Addresses the "does this scale?" question: the main experiments use small-K medical
and CIFAR-10H settings; here we validate the ambiguity-aware calibrators on the
1000-class ImageNet regime, where each validation image is genuinely ambiguous (a
single top-1 label is a lossy summary of a multi-label scene). Two real datasets are
supported, both providing a per-image SOFT target plus a single VOTED label:

  * ImageNet-ReaL  [Beyer et al. 2020, "Are we done with ImageNet?"]
        Each ILSVRC-2012 val image is re-assessed and assigned a SET of valid labels.
        soft target  = normalized indicator over the ReaL label set (captures the
                       genuine ambiguity / multi-label nature of the image),
        voted label  = original single ILSVRC top-1 label.
        Logits come from any pretrained timm/torchvision ImageNet classifier — NO
        training is required (the model already predicts the 1000 classes). K=1000.

  * ImageNet-16H  [Steyvers, Kerrigan et al.]
        Human probabilistic labels over K=16 classes collected at several image-noise
        levels; soft target = the human label distribution, voted = its argmax.
        K=16.

Unlike run_multireader.py (which aggregates ragged per-image reader arrays into a soft
distribution), the ImageNet soft targets are supplied directly by the dataset (a label
SET for ReaL, a human histogram for 16H), so Stage B consumes (soft, voted) directly.

Two-stage design so the calibration logic is verifiable without the (large) ImageNet
val data or a pretrained model download:

  Stage A (dataset+model specific): produce a prepared .npz with
      logits_cal (Ncal,K), logits_test (Nt,K),
      soft_cal (Ncal,K), soft_test (Nt,K),
      voted_cal (Ncal,), voted_test (Nt,), n_classes.
    Use the loader stubs below; they mirror the backbone/extract_logits pattern of
    run_cifar10h.py but skip fine-tuning (ImageNet models are pretrained on the 1000
    classes). NOT runnable in this sandbox (no ImageNet val images / model weights).

  Stage B (generic, runnable here): calibrate + evaluate from those arrays.

GATE: at K=1000 (ReaL) the full K*K matrix methods — Dirichlet (1M params) and Vector
Scaling (K temperatures, plus its per-class optimisation) — are infeasible / overfit on
a calibration split, so we turn them OFF for large K (>50) and report only the scalable
calibrators (TS, LS-TS, SLTS, MCTS S=1, IR-Soft). They remain available for ImageNet-16H
(K=16). See `_DIRICHLET_VS_K_LIMIT` below.

USAGE
-----
  python run_imagenet.py --demo                       # synthetic K=50 multi-label sets; verifies harness
  python run_imagenet.py --data prepared/real_vit.npz --name real_vit_b16
  python run_imagenet.py --dataset real     --imagenet-dir /data/imagenet --arch resnet50
  python run_imagenet.py --dataset 16h      --imagenet16h-dir /data/imagenet16h --noise-level 110
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    TemperatureScaling, DirichletCalibration, SoftLabelTS, MonteCarloTS,
    VectorScaling, LabelSmoothTS, SoftIsotonicRegression, apply_parametric,
)
from metrics import compute_all_metrics, annotation_entropy, print_results_table


# Above this many classes, K^2 (Dirichlet) and per-class (VS) calibrators are
# infeasible / overfit on an ImageNet-scale calibration split, so we gate them off.
_DIRICHLET_VS_K_LIMIT = 50


# ──────────────────────────────────────────────────────────────────────────────
# Stage B: calibrate + evaluate  (generic, reuses calibration.py + metrics.py)
# ──────────────────────────────────────────────────────────────────────────────

def run(logits_cal, soft_cal, voted_cal, logits_test, soft_test, voted_test,
        K, name, n_bins=15):
    """Fit the paper's calibrators and report the standard metric table.

    Parameters
    ----------
    logits_cal/test : (N,K) pre-softmax logits from a pretrained ImageNet model.
    soft_cal/test   : (N,K) per-image soft target (ReaL label-set indicator,
                      normalized, or the ImageNet-16H human distribution).
    voted_cal/test  : (N,) single voted label (ILSVRC top-1 for ReaL).
    K               : number of classes (1000 for ReaL, 16 for 16H).
    """
    soft_cal  = np.asarray(soft_cal, np.float32)
    soft_te   = np.asarray(soft_test, np.float32)
    voted_cal = np.asarray(voted_cal, np.int64)
    voted_te  = np.asarray(voted_test, np.int64)

    lc = torch.tensor(logits_cal, dtype=torch.float32)
    lt = np.asarray(logits_test, np.float32)
    yh = torch.tensor(voted_cal, dtype=torch.long)
    ys = torch.tensor(soft_cal, dtype=torch.float32)
    probs_cal = torch.softmax(lc, 1).numpy()
    probs_te  = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()

    big_K = K > _DIRICHLET_VS_K_LIMIT
    acc = float((lt.argmax(1) == voted_te).mean())
    print(f"\n[{name}]  K={K}  n_cal={len(voted_cal)}  n_test={len(voted_te)}  "
          f"top1-acc(test)={acc*100:.2f}%  "
          f"mean H_test={annotation_entropy(soft_te).mean():.3f}  "
          f"{'(K large: Dirichlet/VS GATED OFF)' if big_K else ''}")

    # voted-label baselines
    ts   = TemperatureScaling().fit(lc, yh)
    lsts = LabelSmoothTS().fit(lc, yh)                  # annotation-free, global eps
    # ambiguity-aware (soft / sampled-annotation targets)
    slts  = SoftLabelTS().fit(lc, ys)
    mcts1 = MonteCarloTS(n_samples=1).fit(lc, ys)       # single sampled label per image
    ir    = SoftIsotonicRegression().fit(probs_cal, soft_cal)

    para = [("Uncalibrated", probs_te),
            ("TS",       apply_parametric(ts,    lt)),
            ("LS-TS",    apply_parametric(lsts,  lt)),
            ("SLTS",     apply_parametric(slts,  lt)),
            ("MCTS S=1", apply_parametric(mcts1, lt))]

    # K^2-parameter calibrators: only when K is small enough to be feasible.
    if not big_K:
        vs   = VectorScaling(K).fit(lc, ys)
        dc_s = DirichletCalibration(K).fit_soft(lc, ys)
        para += [("VS",             apply_parametric(vs,   lt)),
                 ("Dirichlet-Soft", apply_parametric(dc_s, lt))]
    else:
        print(f"  [gate] K={K} > {_DIRICHLET_VS_K_LIMIT}: skipping Dirichlet-Soft "
              f"({K*K} W params) and Vector Scaling — infeasible at ImageNet scale.")

    results = []
    for nm, p in para:
        r = compute_all_metrics(p, voted_te, soft_te, n_bins=n_bins, name=nm)
        r["temperature"] = {"TS": ts.T, "SLTS": slts.T,
                            "MCTS S=1": mcts1.T, "LS-TS": lsts.T}.get(nm)
        results.append(r)

    # IR-Soft (non-parametric, top-class confidence remap)
    cal_conf, pred = ir.calibrate(probs_te)
    pir = np.zeros_like(probs_te)
    for i in range(len(probs_te)):
        pir[i, pred[i]] = cal_conf[i]
        rest = np.ones(K) * (1 - cal_conf[i]) / (K - 1); rest[pred[i]] = 0
        pir[i] += rest
    r = compute_all_metrics(pir, voted_te, soft_te, n_bins=n_bins, name="IR-Soft")
    r["temperature"] = None
    results.append(r)

    print_results_table(results)
    return {"name": name, "n_classes": K, "n_cal": int(len(voted_cal)),
            "n_test": int(len(voted_te)), "top1_acc": acc,
            "dirichlet_vs_gated": bool(big_K), "main_results": results}


# ──────────────────────────────────────────────────────────────────────────────
# Stage A loaders (STUBS — fill in on a machine with the data + a pretrained model).
# Each must return (logits_cal, soft_cal, voted_cal, logits_test, soft_test,
# voted_test, K). The model/logit-extraction mirrors run_cifar10h.py exactly; the
# crucial difference is that the model is PRETRAINED on the 1000 ImageNet classes,
# so there is NO training step — we only forward-pass the val images.
# ──────────────────────────────────────────────────────────────────────────────

def _build_pretrained(arch="resnet50", num_classes=1000, device=None):
    """Return a pretrained ImageNet classifier (no fine-tuning) + its eval transform.

    Mirrors build_model() in run_cifar10h.py but keeps the original 1000-way head.
    arch: any torchvision name ('resnet50','vit_b16',...) or a timm model id.
    """
    import torchvision
    import torchvision.transforms as T
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if arch == "resnet50":
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=w)
    elif arch == "vit_b16":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True,
                                  num_classes=num_classes)
    else:  # try torchvision by name, else timm
        try:
            fn = getattr(torchvision.models, arch)
            model = fn(weights="IMAGENET1K_V1")
        except AttributeError:
            import timm
            model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    return model.to(device).eval(), transform


def _extract_logits(model, dataset, device=None, batch_size=128, cache_path=None):
    """Forward-pass val images -> (N,K) logits. Caches to disk. Mirrors
    extract_logits() in run_cifar10h.py."""
    from torch.utils.data import DataLoader
    if cache_path and Path(cache_path).exists():
        return np.load(cache_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    out = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            out.append(model(x.to(device)).cpu().numpy())
    logits = np.concatenate(out, 0)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, logits)
    return logits


def load_imagenet_real(imagenet_dir, real_json=None, arch="resnet50",
                       cache_dir="./cache", seed=42):
    """ImageNet-ReaL [Beyer et al. 2020, "Are we done with ImageNet?"].

    Each ILSVRC-2012 validation image is re-assessed with a SET of valid labels
    (Reassessed Labels). We turn that set into a per-image SOFT target (a normalized
    indicator over the set — uniform mass over the valid labels, which captures the
    multi-label ambiguity of the image), and keep the original ILSVRC top-1 as the
    VOTED label. Logits come from a PRETRAINED 1000-class model (no training).
    NOT runnable in this sandbox (needs the ~6 GB val images + model weights);
    verified for structure via py_compile and the synthetic --demo path.

    Requires:
      * the 50000 ImageNet val images laid out for torchvision.datasets.ImageFolder
        (or ImageNet) at `imagenet_dir`,
      * the ReaL label file `real.json` (a length-50000 list of label lists; from
        https://github.com/google-research/reassessed-imagenet). Images with an empty
        ReaL set are dropped (Beyer et al. exclude them from ReaL accuracy).
    """
    try:
        import torchvision
    except ImportError as e:
        raise ImportError("ImageNet-ReaL loader needs torchvision (+ optionally timm) "
                          "and the ILSVRC-2012 val images on disk.") from e
    real_json = real_json or str(Path(imagenet_dir) / "real.json")
    if not Path(real_json).exists():
        raise ImportError(
            "ImageNet-ReaL labels not found. Download real.json from "
            "https://github.com/google-research/reassessed-imagenet and pass it via "
            "--real-json (it is a length-50000 list of valid-label lists, aligned to "
            "the sorted ILSVRC val image order).")
    K = 1000

    # val images in the canonical sorted order that the ReaL labels are aligned to.
    model, transform = _build_pretrained(arch, num_classes=K)
    valset = torchvision.datasets.ImageNet(imagenet_dir, split="val", transform=transform)
    orig_labels = np.array([y for _, y in valset.samples], np.int64)  # ILSVRC top-1

    real_sets = json.load(open(real_json))                            # list of lists
    assert len(real_sets) == len(orig_labels), "ReaL/val length mismatch"

    logits_all = _extract_logits(model, valset, cache_path=str(
        Path(cache_dir) / f"imagenet_val_logits_{arch}.npy"))

    keep = [i for i, s in enumerate(real_sets) if len(s) > 0]         # non-empty ReaL set
    soft = np.zeros((len(keep), K), np.float32)
    voted = np.zeros(len(keep), np.int64)
    for j, i in enumerate(keep):
        s = real_sets[i]
        soft[j, s] = 1.0 / len(s)                                     # uniform over valid set
        voted[j] = orig_labels[i]                                     # original top-1
    logits = logits_all[keep]

    # 50/50 calibration / test split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keep)); h = len(keep) // 2
    cal, te = perm[:h], perm[h:]
    return (logits[cal], soft[cal], voted[cal],
            logits[te],  soft[te],  voted[te], K)


def load_imagenet16h(imagenet16h_dir, noise_level=110, arch="resnet50",
                     cache_dir="./cache", seed=42):
    """ImageNet-16H [Steyvers, Kerrigan et al., "Bayesian modeling of human-AI
    complementary ..."]: K=16 superordinate classes, with human probabilistic labels
    collected on phase-noise-corrupted images at several noise levels.

    soft target  = the human label distribution over the 16 classes for each image
                   (the genuine per-image annotator histogram),
    voted label  = its argmax (the human-consensus class).
    Logits come from a model whose 1000-way ImageNet outputs are aggregated into the
    16 superordinate classes (or a 16-way head); mirror run_cifar10h.py's extract step.
    NOT runnable here (needs the ImageNet-16H images + human-label CSVs).

    Requires the ImageNet-16H release: per-image human-judgment CSVs (one row per
    human trial) and the noised image folders, under `imagenet16h_dir`.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("ImageNet-16H loader needs `pip install pandas` and the "
                          "ImageNet-16H human-label CSVs + image folders on disk.") from e
    root = Path(imagenet16h_dir)
    csv = root / f"human_labels_noise{noise_level}.csv"
    if not csv.exists():
        raise ImportError(
            f"ImageNet-16H labels for noise level {noise_level} not found at {csv}. "
            "Obtain the ImageNet-16H release (human-judgment CSVs with columns "
            "image_id, true_class, participant_classification) and the matching noised "
            "image folders, then point --imagenet16h-dir at it.")
    K = 16

    df = pd.read_csv(csv)
    classes = sorted(df["true_class"].unique())
    assert len(classes) == K, f"expected 16 classes, got {len(classes)}"
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    # Per-image human label distribution (soft) + voted argmax.
    image_ids, soft_rows = [], []
    for img_id, g in df.groupby("image_id"):
        counts = np.zeros(K, np.float32)
        for c in g["participant_classification"]:
            counts[cls_to_idx[c]] += 1
        soft_rows.append(counts / counts.sum())
        image_ids.append(img_id)
    soft = np.stack(soft_rows)
    voted = soft.argmax(1).astype(np.int64)

    # Forward-pass the matching (noised) images through a pretrained model whose
    # outputs are mapped to the 16 superordinate classes. Mirror run_cifar10h.py.
    from torch.utils.data import Dataset
    from PIL import Image
    model, transform = _build_pretrained(arch, num_classes=K)

    class _ImgDS(Dataset):
        def __init__(self, ids): self.ids = ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            p = root / "images" / f"noise{noise_level}" / f"{self.ids[i]}.png"
            return transform(Image.open(p).convert("RGB")),
    logits = _extract_logits(model, _ImgDS(image_ids), cache_path=str(
        Path(cache_dir) / f"imagenet16h_logits_{arch}_noise{noise_level}.npy"))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(image_ids)); h = len(image_ids) // 2
    cal, te = perm[:h], perm[h:]
    return (logits[cal], soft[cal], voted[cal],
            logits[te],  soft[te],  voted[te], K)


LOADERS = {"real": load_imagenet_real, "16h": load_imagenet16h}


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic demo: ImageNet-style MULTI-LABEL sets at moderate K (no data / model).
# Mirrors the ReaL regime: each image gets a SET of valid labels => a soft target.
# ──────────────────────────────────────────────────────────────────────────────

def make_demo(seed=0, K=50, n=8000):
    """Each image has a true class plus, with per-image difficulty d_i, one or more
    EXTRA valid labels (a ReaL-style label SET). The soft target is the uniform
    distribution over that set; the voted label is the true class. A pretrained-style
    'model' puts logit mass on the set members but is overconfident on the easy cases,
    so the uncalibrated model should be miscalibrated against the soft targets."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)                       # true / voted class
    d = rng.beta(1.5, 4.0, size=n)                        # per-image ambiguity in (0,1)
    soft = np.zeros((n, K), np.float32)
    logits = rng.normal(0, 0.3, size=(n, K)).astype(np.float32)
    for i in range(n):
        label_set = {int(y[i])}
        # number of extra valid labels grows with difficulty
        n_extra = rng.binomial(3, d[i])
        while len(label_set) < 1 + n_extra:
            label_set.add(int(rng.integers(0, K)))
        members = list(label_set)
        soft[i, members] = 1.0 / len(members)
        # model: mass on every set member, scaled down on ambiguous images (overconfident)
        logits[i, members] += 3.0 * (1 - 0.5 * d[i])
    idx = rng.permutation(n); h = n // 2
    cal, te = idx[:h], idx[h:]
    return (logits[cal], soft[cal], y[cal],
            logits[te],  soft[te],  y[te], K)


def main():
    ap = argparse.ArgumentParser(description="ImageNet-scale ambiguous-GT calibration")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic K=50 multi-label sets; verifies the harness")
    ap.add_argument("--dataset", choices=list(LOADERS),
                    help="real dataset (needs ImageNet val images + a pretrained model)")
    ap.add_argument("--data", help="prepared .npz (logits_cal, soft_cal, voted_cal, "
                                    "logits_test, soft_test, voted_test, n_classes)")
    ap.add_argument("--imagenet-dir", help="ILSVRC-2012 dir for --dataset real")
    ap.add_argument("--real-json", help="ReaL real.json for --dataset real")
    ap.add_argument("--imagenet16h-dir", help="ImageNet-16H dir for --dataset 16h")
    ap.add_argument("--noise-level", type=int, default=110, help="ImageNet-16H noise level")
    ap.add_argument("--arch", default="resnet50", help="pretrained backbone")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--name", default="imagenet")
    ap.add_argument("--n-bins", type=int, default=15)
    args = ap.parse_args()

    if args.demo:
        lc, sc, vc, lt, st, vt, K = make_demo()
        name = "demo_imagenet"
    elif args.data:
        z = np.load(args.data, allow_pickle=True)
        lc, sc, vc = z["logits_cal"], z["soft_cal"], z["voted_cal"]
        lt, st, vt = z["logits_test"], z["soft_test"], z["voted_test"]
        K = int(z["n_classes"]); name = args.name
    elif args.dataset == "real":
        lc, sc, vc, lt, st, vt, K = load_imagenet_real(
            args.imagenet_dir, args.real_json, arch=args.arch, cache_dir=args.cache_dir)
        name = args.name if args.name != "imagenet" else f"real_{args.arch}"
    elif args.dataset == "16h":
        lc, sc, vc, lt, st, vt, K = load_imagenet16h(
            args.imagenet16h_dir, noise_level=args.noise_level, arch=args.arch,
            cache_dir=args.cache_dir)
        name = args.name if args.name != "imagenet" else f"16h_{args.arch}_n{args.noise_level}"
    else:
        ap.error("specify --demo, --data, or --dataset")

    out = run(lc, sc, vc, lt, st, vt, K, name, n_bins=args.n_bins)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    p = Path(args.results_dir) / f"imagenet_{name}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
