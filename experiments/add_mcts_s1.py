"""
Compute MCTS (S=1) results for all 8 settings and append to existing JSON files.

This script replicates the exact data splits from the original experiment scripts
(same seed, same logic) but only adds MCTS(n_samples=1) to the results.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibration import MonteCarloTS, apply_parametric
from metrics import compute_all_metrics

CACHE = Path(__file__).parent / "cache"
RESULTS = Path(__file__).parent / "results"

# ── Helpers ────────────────────────────────────────────────────────────────────

def mcts_s1_row(logits_cal, ys_cal, logits_te, yh_te, ys_te, n_bins=15):
    """Fit MCTS(S=1) and return a metrics dict."""
    ys_cal_t = torch.tensor(ys_cal, dtype=torch.float32)
    mcts1 = MonteCarloTS(n_samples=1, seed=42).fit(
        torch.tensor(logits_cal, dtype=torch.float32), ys_cal_t
    )
    p = apply_parametric(mcts1, torch.tensor(logits_te, dtype=torch.float32))
    r = compute_all_metrics(p, yh_te, ys_te, n_bins=n_bins, name="MCTS S=1 (ours)")
    r["temperature"] = float(mcts1.T)
    return r


def update_json(path, new_row, insert_after="MCTS (ours)"):
    """Insert new_row after insert_after in main_results list."""
    data = json.load(open(path))
    mr = data["main_results"]
    # Check if already exists
    if any(r["name"] == new_row["name"] for r in mr):
        print(f"  {new_row['name']} already in {path.name}, skipping")
        return
    idx = next((i for i, r in enumerate(mr) if r["name"] == insert_after), None)
    if idx is not None:
        mr.insert(idx + 1, new_row)
    else:
        mr.append(new_row)
    json.dump(data, open(path, "w"), indent=2)
    print(f"  Added {new_row['name']} (T={new_row['temperature']:.4f}) → {path.name}")


# ── CIFAR-10H ──────────────────────────────────────────────────────────────────

def add_cifar10h(arch):
    logits_all = np.load(CACHE / f"logits_test_{arch}.npy")
    cifar10h   = np.load(CACHE / "cifar10h-probs.npy")          # (10000, 10)
    hard_labels = cifar10h.argmax(axis=1)

    rng = np.random.default_rng(42)
    idx = np.arange(10000)
    cal_mask = np.zeros(10000, dtype=bool)
    for c in range(10):
        c_idx  = idx[hard_labels == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True
    idx_cal = idx[cal_mask];  idx_te = idx[~cal_mask]

    row = mcts_s1_row(
        logits_all[idx_cal], cifar10h[idx_cal],
        logits_all[idx_te],  hard_labels[idx_te], cifar10h[idx_te],
    )
    arch_tag = "resnet50" if arch == "resnet50" else "vit_b16"
    update_json(RESULTS / f"cifar10h_results_{arch_tag}.json", row)


# ── ChaosNLI ──────────────────────────────────────────────────────────────────

def _parse_chaosnli():
    """Parse combined SNLI+MNLI ChaosNLI data (mirrors run_chaosnli.py logic)."""
    import json as _json
    base = CACHE / "chaosNLI_v1.0"
    all_soft = []
    for fname in ["chaosNLI_snli.jsonl", "chaosNLI_mnli_m.jsonl"]:
        path = base / fname
        for line in open(path):
            ex = _json.loads(line.strip())
            dist = ex["label_dist"]  # already a list [ent, neu, con]
            all_soft.append(np.array(dist, dtype=np.float32))
    soft = np.array(all_soft)
    hard = soft.argmax(axis=1)
    return soft, hard


def add_chaosnli(arch):
    logits_all = np.load(CACHE / f"logits_chaosnli_combined_{arch}.npy")
    soft_labels, hard_labels = _parse_chaosnli()
    N = len(soft_labels)

    rng = np.random.default_rng(42)
    idx = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
    for c in range(3):
        c_idx  = idx[hard_labels == c]
        chosen = rng.choice(c_idx, size=len(c_idx) // 2, replace=False)
        cal_mask[chosen] = True
    idx_cal = idx[cal_mask];  idx_te = idx[~cal_mask]

    row = mcts_s1_row(
        logits_all[idx_cal], soft_labels[idx_cal],
        logits_all[idx_te],  hard_labels[idx_te], soft_labels[idx_te],
    )
    arch_tag = "roberta_large" if "roberta" in arch else "deberta_v3"
    update_json(RESULTS / f"chaosnli_combined_results_{arch_tag}.json", row)


# ── ISIC 2019 ─────────────────────────────────────────────────────────────────

ISIC_CONFUSION = np.array([
    [0.73, 0.14, 0.02, 0.03, 0.08, 0.00, 0.00, 0.00],  # MEL
    [0.15, 0.76, 0.01, 0.01, 0.06, 0.01, 0.00, 0.00],  # NV
    [0.02, 0.01, 0.81, 0.05, 0.07, 0.01, 0.01, 0.02],  # BCC
    [0.03, 0.01, 0.04, 0.65, 0.11, 0.00, 0.00, 0.16],  # AK
    [0.12, 0.05, 0.03, 0.10, 0.62, 0.00, 0.00, 0.08],  # BKL
    [0.01, 0.02, 0.02, 0.01, 0.02, 0.87, 0.03, 0.02],  # DF
    [0.00, 0.01, 0.02, 0.01, 0.01, 0.02, 0.91, 0.02],  # VL
    [0.01, 0.01, 0.03, 0.18, 0.09, 0.00, 0.01, 0.67],  # SCC
], dtype=np.float64)

DERM_CONFUSION = np.array([
    [0.62, 0.07, 0.16, 0.02, 0.07, 0.05, 0.01],
    [0.07, 0.73, 0.09, 0.03, 0.04, 0.03, 0.01],
    [0.12, 0.06, 0.62, 0.05, 0.08, 0.06, 0.01],
    [0.02, 0.03, 0.04, 0.83, 0.03, 0.04, 0.01],
    [0.04, 0.04, 0.07, 0.02, 0.63, 0.19, 0.01],
    [0.03, 0.02, 0.06, 0.04, 0.14, 0.70, 0.01],
    [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.92],
], dtype=np.float64)


def _gen_soft(hard_labels, confusion, n_annotators, seed):
    rng  = np.random.default_rng(seed)
    N, K = len(hard_labels), confusion.shape[0]
    soft = np.zeros((N, K), dtype=np.float32)
    for i, y in enumerate(hard_labels):
        anns = rng.choice(K, size=n_annotators, p=confusion[y])
        for a in anns:
            soft[i, a] += 1.0
        soft[i] /= n_annotators
    return soft


def add_isic(arch):
    arch_tag = "efficientnet_b4" if "enet" in arch or "efficientnet" in arch else "vit_s16"
    logits_val  = np.load(CACHE / f"isic2019_logits_val_{arch_tag}.npy")
    logits_test = np.load(CACHE / f"isic2019_logits_test_{arch_tag}.npy")
    val_labels  = np.load(CACHE / "isic2019_labels_val.npy")
    test_labels = np.load(CACHE / "isic2019_labels_test.npy")

    ys_val  = _gen_soft(val_labels,  ISIC_CONFUSION, n_annotators=9, seed=42)
    ys_test = _gen_soft(test_labels, ISIC_CONFUSION, n_annotators=9, seed=43)

    row = mcts_s1_row(
        logits_val, ys_val,
        logits_test, test_labels, ys_test,
    )
    update_json(RESULTS / f"isic2019_results_{arch_tag}.json", row)


def add_dermamnist(arch):
    arch_tag = arch
    logits_val  = np.load(CACHE / f"dermamnist_logits_val_{arch_tag}.npy")
    logits_test = np.load(CACHE / f"dermamnist_logits_test_{arch_tag}.npy")
    d = np.load(CACHE / "dermamnist.npz")
    val_labels  = d["val_labels"].squeeze()
    test_labels = d["test_labels"].squeeze()

    ys_val  = _gen_soft(val_labels,  DERM_CONFUSION, n_annotators=5, seed=42)
    ys_test = _gen_soft(test_labels, DERM_CONFUSION, n_annotators=5, seed=43)

    row = mcts_s1_row(
        logits_val, ys_val,
        logits_test, test_labels, ys_test,
    )
    update_json(RESULTS / f"dermamnist_results_{arch_tag}.json", row)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== CIFAR-10H ===")
    add_cifar10h("resnet50");  add_cifar10h("vit_b16")

    print("\n=== ChaosNLI ===")
    add_chaosnli("roberta_large");  add_chaosnli("deberta_v3")

    print("\n=== ISIC 2019 ===")
    add_isic("efficientnet_b4");  add_isic("vit_s16")

    print("\n=== DermaMNIST ===")
    add_dermamnist("resnet18");  add_dermamnist("vit_s16")

    print("\nDone.")
