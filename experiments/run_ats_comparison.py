"""
ATS Comparison: Adaptive Temperature Scaling vs. global baselines.

Compares on all 8 benchmark settings (4 datasets × 2 architectures):
  - TS         : global T, voted-label target         (annotation-free)
  - LS-TS      : global T, smoothed voted-label target (annotation-free)
  - ATS-Hard   : per-instance T, voted-label target   (annotation-free)
  - SLTS       : global T, soft-label target           (requires annotators)
  - ATS-Soft   : per-instance T, soft-label target     (requires annotators)

Usage
-----
    cd experiments/
    python run_ats_comparison.py [--cache-dir ./cache] [--seed 42] [--n-bins 15]
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
    SoftLabelTS,
    AdaptiveTempScaling,
    AdaptiveSoftLabelTS,
    apply_parametric,
)
from metrics import compute_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (identical split logic to other experiment scripts)
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10h(cache_dir, arch, seed):
    logits_all = np.load(Path(cache_dir) / f"logits_test_{arch}.npy")
    soft_all   = np.load(Path(cache_dir) / "cifar10h-probs.npy").astype(np.float32)
    hard_all   = soft_all.argmax(axis=1)
    N, K       = logits_all.shape
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(K):
        c_idx = idx[hard_all == c]
        cal_mask[rng.choice(c_idx, size=len(c_idx) // 2, replace=False)] = True
    ic, it = idx[cal_mask], idx[~cal_mask]
    return (torch.tensor(logits_all[ic], dtype=torch.float32),
            torch.tensor(logits_all[it], dtype=torch.float32),
            hard_all[ic], hard_all[it], soft_all[ic], soft_all[it])


def load_chaosnli(cache_dir, arch, seed):
    import json as _j
    data_dir = Path(cache_dir) / "chaosNLI_v1.0"
    raw = []
    for fname in ("chaosNLI_snli.jsonl", "chaosNLI_mnli_m.jsonl"):
        with open(data_dir / fname) as f:
            for line in f:
                raw.append(_j.loads(line.strip()))
    soft_list, hard_list = [], []
    for d in raw:
        ld = d.get("label_dist")
        if isinstance(ld, list):
            soft = np.array(ld, dtype=np.float32)
        else:
            cnt = d.get("label_counter", {})
            c   = np.array([cnt.get("e", cnt.get("entailment", 0)),
                            cnt.get("n", cnt.get("neutral", 0)),
                            cnt.get("c", cnt.get("contradiction", 0))], dtype=np.float32)
            soft = c / c.sum() if c.sum() > 0 else np.ones(3) / 3
        soft_list.append(soft)
        hard_list.append(int(soft.argmax()))
    soft_all = np.array(soft_list, dtype=np.float32)
    hard_all = np.array(hard_list, dtype=np.int64)
    logits_all = np.load(Path(cache_dir) / f"logits_chaosnli_combined_{arch}.npy")
    N, K = logits_all.shape
    rng  = np.random.default_rng(seed)
    idx  = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(K):
        c_idx = idx[hard_all == c]
        cal_mask[rng.choice(c_idx, size=len(c_idx) // 2, replace=False)] = True
    ic, it = idx[cal_mask], idx[~cal_mask]
    return (torch.tensor(logits_all[ic], dtype=torch.float32),
            torch.tensor(logits_all[it], dtype=torch.float32),
            hard_all[ic], hard_all[it], soft_all[ic], soft_all[it])


def load_dermamnist(cache_dir, arch, seed):
    logits_all = np.load(Path(cache_dir) / f"logits_dermamnist_{arch}.npy")
    soft_all   = np.load(Path(cache_dir) / f"soft_labels_dermamnist_{arch}.npy").astype(np.float32)
    hard_all   = soft_all.argmax(axis=1)
    N, K       = logits_all.shape
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(K):
        c_idx = idx[hard_all == c]
        if len(c_idx) > 0:
            cal_mask[rng.choice(c_idx, size=max(1, len(c_idx) // 2), replace=False)] = True
    ic, it = idx[cal_mask], idx[~cal_mask]
    return (torch.tensor(logits_all[ic], dtype=torch.float32),
            torch.tensor(logits_all[it], dtype=torch.float32),
            hard_all[ic], hard_all[it], soft_all[ic], soft_all[it])


def load_isic2019(cache_dir, arch, seed):
    logits_all = np.load(Path(cache_dir) / f"logits_isic2019_{arch}.npy")
    soft_all   = np.load(Path(cache_dir) / f"soft_labels_isic2019_{arch}.npy").astype(np.float32)
    hard_all   = soft_all.argmax(axis=1)
    N, K       = logits_all.shape
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(K):
        c_idx = idx[hard_all == c]
        if len(c_idx) > 0:
            cal_mask[rng.choice(c_idx, size=max(1, len(c_idx) // 2), replace=False)] = True
    ic, it = idx[cal_mask], idx[~cal_mask]
    return (torch.tensor(logits_all[ic], dtype=torch.float32),
            torch.tensor(logits_all[it], dtype=torch.float32),
            hard_all[ic], hard_all[it], soft_all[ic], soft_all[it])


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one(lc, lt, yh_c, yh_t, ys_c, ys_t, n_bins, label):
    yh_c_t  = torch.tensor(yh_c, dtype=torch.long)
    ys_c_t  = torch.tensor(ys_c, dtype=torch.float32)

    methods = [
        ("TS",         TemperatureScaling().fit(lc, yh_c_t),             False),
        ("LS-TS",      LabelSmoothTS().fit(lc, yh_c_t),                  False),
        ("ATS-Hard",   AdaptiveTempScaling().fit(lc, yh_c_t),            True),
        ("SLTS",       SoftLabelTS().fit(lc, ys_c_t),                    False),
        ("ATS-Soft",   AdaptiveSoftLabelTS().fit(lc, ys_c_t),            True),
    ]

    print(f"\n  {label}")
    print(f"  {'Method':<12} {'T_mean':>7}  {'ECE%':>6}  {'Brier':>7}  {'NLL':>7}")
    print(f"  {'-'*50}")

    results = {}
    for name, model, is_adaptive in methods:
        p = apply_parametric(model, lt.numpy())
        m = compute_all_metrics(p, yh_t, ys_t, n_bins=n_bins, name=name)
        T_str = f"{model.mean_T(lt):6.3f}" if is_adaptive else f"{model.T:6.3f}"
        print(f"  {name:<12} {T_str}  "
              f"{m['ece_sampled']*100:6.2f}%  "
              f"{m['brier_sampled']:.4f}  "
              f"{m['nll_sampled']:.4f}")
        results[name] = {
            "T_mean": model.mean_T(lt) if is_adaptive else model.T,
            "ece":    m["ece_sampled"],
            "brier":  m["brier_sampled"],
            "nll":    m["nll_sampled"],
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir",   default="./cache")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    configs = [
        ("cifar10h",  "resnet50",        "CIFAR-10H ResNet-50",       load_cifar10h),
        ("cifar10h",  "vit_b16",         "CIFAR-10H ViT-B/16",        load_cifar10h),
        ("chaosnli",  "roberta_large",   "ChaosNLI RoBERTa-L",        load_chaosnli),
        ("chaosnli",  "deberta_v3",      "ChaosNLI DeBERTa-v3",       load_chaosnli),
        ("dermamnist","resnet18",         "DermaMNIST ResNet-18",      load_dermamnist),
        ("dermamnist","vit_s16",          "DermaMNIST ViT-S/16",       load_dermamnist),
        ("isic2019",  "efficientnet_b4", "ISIC 2019 ENet-B4",         load_isic2019),
        ("isic2019",  "vit_s16",         "ISIC 2019 ViT-S/16",        load_isic2019),
    ]

    all_results = {}
    for dataset, arch, label, loader in configs:
        print(f"\n[{label}] Loading…")
        try:
            data = loader(args.cache_dir, arch, args.seed)
        except FileNotFoundError as e:
            print(f"  SKIP (file not found): {e}")
            continue
        key = f"{dataset}_{arch}"
        all_results[key] = run_one(*data, n_bins=args.n_bins, label=label)

    out = Path(args.results_dir) / "ats_comparison.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
