"""
LS-TS Ablation: compare annotation-free smoothing strategies.

Compares four variants on CIFAR-10H and ChaosNLI (real annotator distributions):
  - TS          : baseline (voted labels, no smoothing)
  - LS-TS       : our method — global ε = mean(1 − f[y*])
  - Fixed-LS    : ablation — fixed ε = 0.1 (data-independent)
  - Ent-LS      : ablation — per-instance ε_i = H(f(x_i)) / log K
  - CC-LS       : ablation — per-class ε_k from class-conditional mean confidence

All methods use the same single-temperature family; only the pseudo-target
construction differs.  Uses cached logits from the main experiments.

Usage
-----
    cd experiments/
    python run_lsts_ablation.py [--cache-dir ./cache] [--results-dir ./results]
                                [--seed 42] [--n-bins 15]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    TemperatureScaling,
    LabelSmoothTS,
    FixedLabelSmoothTS,
    EntropyLabelSmoothTS,
    ClassCondLabelSmoothTS,
    apply_parametric,
)
from metrics import compute_all_metrics, annotation_entropy


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (reuse cached logits + annotation files from main experiments)
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10h(cache_dir: str, arch: str, seed: int):
    """Return (logits_cal, logits_te, yh_cal, yh_te, ys_cal, ys_te)."""
    import json as _json  # local to avoid name clash

    logits_all = np.load(Path(cache_dir) / f"logits_test_{arch}.npy")
    soft_all   = np.load(Path(cache_dir) / "cifar10h-probs.npy").astype(np.float32)
    hard_all   = soft_all.argmax(axis=1)
    N, K       = logits_all.shape

    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(K):
        c_idx  = idx[hard_all == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True

    ic, it = idx[cal_mask], idx[~cal_mask]
    return (
        torch.tensor(logits_all[ic], dtype=torch.float32),
        torch.tensor(logits_all[it], dtype=torch.float32),
        hard_all[ic], hard_all[it],
        soft_all[ic], soft_all[it],
    )


def load_chaosnli(cache_dir: str, arch: str, seed: int):
    """Return (logits_cal, logits_te, yh_cal, yh_te, ys_cal, ys_te)."""
    import json as _json

    # Load annotation data
    data_dir = Path(cache_dir) / "chaosNLI_v1.0"
    raw = []
    for fname in ("chaosNLI_snli.jsonl", "chaosNLI_mnli_m.jsonl"):
        with open(data_dir / fname) as f:
            for line in f:
                raw.append(_json.loads(line.strip()))

    # Parse soft labels (canonical order: entailment, neutral, contradiction)
    soft_list, hard_list = [], []
    for d in raw:
        ld = d.get("label_dist")
        if isinstance(ld, list):
            # label_dist is already [e, n, c] probabilities
            soft = np.array(ld, dtype=np.float32)
        else:
            # fallback: label_counter dict {e:N, n:N, c:N}
            counter = d.get("label_counter", {})
            counts  = np.array([
                counter.get("e", counter.get("entailment", 0)),
                counter.get("n", counter.get("neutral", 0)),
                counter.get("c", counter.get("contradiction", 0)),
            ], dtype=np.float32)
            total = counts.sum()
            soft = counts / total if total > 0 else np.ones(3) / 3
        soft_list.append(soft)
        hard_list.append(int(soft.argmax()))
    soft_all = np.array(soft_list, dtype=np.float32)
    hard_all = np.array(hard_list, dtype=np.int64)

    # Cached logits are already in canonical [e, n, c] order (saved by run_chaosnli.py)
    logits_all = np.load(Path(cache_dir) / f"logits_chaosnli_combined_{arch}.npy")

    N, K = logits_all.shape
    rng  = np.random.default_rng(seed)
    idx  = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(K):
        c_idx  = idx[hard_all == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True

    ic, it = idx[cal_mask], idx[~cal_mask]
    return (
        torch.tensor(logits_all[ic], dtype=torch.float32),
        torch.tensor(logits_all[it], dtype=torch.float32),
        hard_all[ic], hard_all[it],
        soft_all[ic], soft_all[it],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(logits_cal, logits_te, yh_cal, yh_te, ys_cal, ys_te,
                 n_bins: int, label: str):
    yh_cal_t = torch.tensor(yh_cal, dtype=torch.long)
    probs_te  = torch.softmax(logits_te, dim=1).numpy()

    results = {}

    methods = [
        ("TS",        TemperatureScaling().fit(logits_cal, yh_cal_t)),
        ("LS-TS",     LabelSmoothTS().fit(logits_cal, yh_cal_t)),
        ("Fixed-LS (ε=0.1)",  FixedLabelSmoothTS(eps=0.1).fit(logits_cal, yh_cal_t)),
        ("Ent-LS",    EntropyLabelSmoothTS().fit(logits_cal, yh_cal_t)),
        ("CC-LS",     ClassCondLabelSmoothTS().fit(logits_cal, yh_cal_t)),
    ]

    print(f"\n  {label}")
    print(f"  {'Method':<22} T       ECE    Brier   NLL")
    print(f"  {'-'*60}")

    for name, model in methods:
        p = apply_parametric(model, logits_te.numpy())
        m = compute_all_metrics(p, yh_te, ys_te, n_bins=n_bins, name=name)
        T = model.T
        print(f"  {name:<22} {T:5.3f}  "
              f"{m['ece_sampled']*100:5.2f}%  "
              f"{m['brier_sampled']:.4f}  "
              f"{m['nll_sampled']:.4f}")
        results[name] = {
            "T": T,
            "ece": m["ece_sampled"],
            "brier": m["brier_sampled"],
            "nll": m["nll_sampled"],
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir",   default="./cache")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--n-bins",  type=int, default=15)
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    configs = [
        ("cifar10h", "resnet50",      "CIFAR-10H ResNet-50",      load_cifar10h),
        ("cifar10h", "vit_b16",       "CIFAR-10H ViT-B/16",       load_cifar10h),
        ("chaosnli", "roberta_large", "ChaosNLI RoBERTa-L",       load_chaosnli),
        ("chaosnli", "deberta_v3",    "ChaosNLI DeBERTa-v3",      load_chaosnli),
    ]

    for dataset, arch, label, loader in configs:
        print(f"\n[{label}] Loading data…")
        logits_cal, logits_te, yh_cal, yh_te, ys_cal, ys_te = loader(
            args.cache_dir, arch, args.seed
        )
        key = f"{dataset}_{arch}"
        all_results[key] = run_ablation(
            logits_cal, logits_te, yh_cal, yh_te, ys_cal, ys_te,
            n_bins=args.n_bins, label=label,
        )

    out = Path(args.results_dir) / "lsts_ablation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
