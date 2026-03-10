"""
Experiment: Calibration under Ambiguous Ground Truth on ChaosNLI.

ChaosNLI (Nie et al., EMNLP 2020) provides 100 human annotations per example
for subsets of SNLI and MultiNLI dev sets (3-class NLI: entailment, neutral,
contradiction).  This is the NLP analog of CIFAR-10H.

Pipeline
--------
1. Download ChaosNLI annotation files (JSONL) from GitHub.
2. Load a pre-fine-tuned NLI model (RoBERTa-large-MNLI or DeBERTa-v3-base-MNLI).
3. Tokenize premise–hypothesis pairs and extract logits.
4. Split into calibration and test sets (stratified by voted label).
5. Fit all calibration methods on the calibration split.
6. Evaluate voted-label + true-label metrics on the test split.
7. Save results to experiments/results/chaosnli_results.json.

Usage
-----
    python run_chaosnli.py [--arch roberta_large|deberta_v3] [--subset snli|mnli]
                           [--cache-dir ./cache] [--results-dir ./results]
                           [--n-bins 15] [--seed 42] [--device cpu|cuda]

Requirements
------------
    torch transformers datasets requests scikit-learn tqdm numpy
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
from tqdm import tqdm

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from calibration import (
    TemperatureScaling, PlattScaling, DirichletCalibration,
    SoftPlattScaling,
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

CHAOSNLI_DROPBOX_URL = "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip"

CHAOSNLI_FILES = {
    "snli": "chaosNLI_v1.0/chaosNLI_snli.jsonl",
    "mnli": "chaosNLI_v1.0/chaosNLI_mnli_m.jsonl",
}

NLI_LABELS = ["entailment", "neutral", "contradiction"]
N_CLASSES = 3

# Model name mappings
MODEL_CONFIGS = {
    "roberta_large": {
        "hf_name": "roberta-large-mnli",
        "short_name": "RoBERTa-L",
        # RoBERTa-large-MNLI label order: contradiction=0, neutral=1, entailment=2
        "label_map": {"contradiction": 0, "neutral": 1, "entailment": 2},
    },
    "deberta_v3": {
        "hf_name": "cross-encoder/nli-deberta-v3-base",
        "short_name": "DeBERTa-v3",
        # cross-encoder NLI models: contradiction=0, entailment=1, neutral=2
        "label_map": {"contradiction": 0, "entailment": 1, "neutral": 2},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def download_chaosnli(cache_dir: str, subset: str = "snli") -> list[dict]:
    """Download (once) and return ChaosNLI data as list of dicts."""
    import zipfile

    chaosnli_dir = Path(cache_dir) / "chaosNLI_v1.0"
    if not chaosnli_dir.exists():
        zip_path = Path(cache_dir) / "chaosNLI_v1.0.zip"
        if not zip_path.exists():
            print(f"Downloading ChaosNLI from Dropbox → {zip_path}")
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(CHAOSNLI_DROPBOX_URL, zip_path)
        print("Extracting ChaosNLI ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(cache_dir)
        zip_path.unlink()

    path = Path(cache_dir) / CHAOSNLI_FILES[subset]
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"  Loaded {len(data)} examples from ChaosNLI-{subset}")
    return data


def parse_chaosnli(data: list[dict]) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Parse ChaosNLI JSONL data into:
      - examples: list of (premise, hypothesis) tuples
      - soft_labels: (N, 3) annotator distribution [entailment, neutral, contradiction]
      - hard_labels: (N,) majority-voted label indices

    ChaosNLI label_counter keys: 'e' (entailment), 'n' (neutral), 'c' (contradiction)
    We use canonical order: [entailment=0, neutral=1, contradiction=2]
    """
    examples = []
    soft_labels = []

    for item in data:
        # Extract text pair
        if "example" in item:
            ex = item["example"]
            premise = ex.get("premise", ex.get("sentence1", ""))
            hypothesis = ex.get("hypothesis", ex.get("sentence2", ""))
        else:
            premise = item.get("premise", item.get("sentence1", ""))
            hypothesis = item.get("hypothesis", item.get("sentence2", ""))

        examples.append((premise, hypothesis))

        # Extract label counts → soft distribution
        lc = item.get("label_counter", item.get("label_count", {}))
        counts = np.array([
            lc.get("e", lc.get("entailment", 0)),
            lc.get("n", lc.get("neutral", 0)),
            lc.get("c", lc.get("contradiction", 0)),
        ], dtype=np.float32)
        total = counts.sum()
        if total > 0:
            soft_labels.append(counts / total)
        else:
            soft_labels.append(np.ones(3, dtype=np.float32) / 3)

    soft_labels = np.array(soft_labels)  # (N, 3) in [ent, neu, con] order
    hard_labels = soft_labels.argmax(axis=1)  # voted label

    return examples, soft_labels, hard_labels


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def load_nli_model(arch: str, device: str):
    """Load a pre-fine-tuned NLI model from HuggingFace."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    config = MODEL_CONFIGS[arch]
    hf_name = config["hf_name"]
    print(f"  Loading {hf_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name)
    model = model.to(device).eval()

    return model, tokenizer, config


def extract_logits_nli(
    model,
    tokenizer,
    examples: list[tuple[str, str]],
    device: str,
    batch_size: int = 32,
    cache_path: str = None,
    label_map: dict = None,
) -> np.ndarray:
    """
    Extract logits for NLI premise-hypothesis pairs.
    Returns logits in canonical order [entailment, neutral, contradiction].
    """
    if cache_path and Path(cache_path).exists():
        print(f"  Loading cached logits: {cache_path}")
        return np.load(cache_path)

    all_logits = []
    for start in tqdm(range(0, len(examples), batch_size), desc="  extracting logits"):
        batch = examples[start : start + batch_size]
        premises = [p for p, h in batch]
        hypotheses = [h for p, h in batch]

        inputs = tokenizer(
            premises, hypotheses,
            padding=True, truncation=True, max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()  # (B, model_n_classes)

        all_logits.append(logits)

    logits = np.concatenate(all_logits, axis=0)  # (N, model_n_classes)

    # Reorder logits to canonical [entailment, neutral, contradiction]
    if label_map is not None:
        canonical_logits = np.zeros((logits.shape[0], N_CLASSES), dtype=np.float32)
        ent_idx = label_map.get("entailment", 0)
        neu_idx = label_map.get("neutral", 1)
        con_idx = label_map.get("contradiction", 2)
        canonical_logits[:, 0] = logits[:, ent_idx]
        canonical_logits[:, 1] = logits[:, neu_idx]
        canonical_logits[:, 2] = logits[:, con_idx]
        logits = canonical_logits

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, logits)
        print(f"  Logits cached → {cache_path}")

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
    arch        = args.arch
    subset      = args.subset

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading ChaosNLI ({subset}) data ...")
    if subset == "combined":
        raw_data = download_chaosnli(cache_dir, subset="snli") + \
                   download_chaosnli(cache_dir, subset="mnli")
    else:
        raw_data = download_chaosnli(cache_dir, subset=subset)
    examples, soft_labels, hard_labels = parse_chaosnli(raw_data)
    N = len(examples)
    print(f"  Total examples: {N}")
    print(f"  Label distribution: E={np.mean(hard_labels==0):.2%}, "
          f"N={np.mean(hard_labels==1):.2%}, C={np.mean(hard_labels==2):.2%}")
    print(f"  Mean annotation entropy: {annotation_entropy(soft_labels).mean():.4f}")

    # ── 2. Model (skip if logits are cached) ──────────────────────────────────
    logits_cache = os.path.join(cache_dir, f"logits_chaosnli_{subset}_{arch}.npy")
    config = MODEL_CONFIGS[arch]
    if not os.path.exists(logits_cache):
        print(f"\n[2/5] Loading model ({arch}) ...")
        model, tokenizer, _ = load_nli_model(arch, device)
    else:
        print(f"\n[2/5] Logits cache found, skipping model load ({arch})")
        model, tokenizer = None, None

    # ── 3. Logits ─────────────────────────────────────────────────────────────
    print(f"\n[3/5] Extracting logits ...")
    logits_all = extract_logits_nli(
        model, tokenizer, examples, device,
        batch_size=32, cache_path=logits_cache,
        label_map=config["label_map"],
    )

    probs_all = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
    acc = float((logits_all.argmax(1) == hard_labels).mean())
    print(f"  Voted-label accuracy: {acc*100:.2f}%")

    # ── 4. Cal / test split (50/50, stratified by voted label) ────────────────
    print(f"\n[4/5] Splitting into cal / test ...")
    idx = np.arange(N)
    cal_mask = np.zeros(N, dtype=bool)
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
    ys_cal = soft_labels[idx_cal]
    yh_te  = hard_labels[idx_te]
    ys_te  = soft_labels[idx_te]

    yh_cal_t = torch.tensor(yh_cal, dtype=torch.long)
    ys_cal_t = torch.tensor(ys_cal, dtype=torch.float32)

    print(f"  Cal n={len(idx_cal)}, Test n={len(idx_te)}")
    print(f"  Mean annotation entropy (test): {annotation_entropy(ys_te).mean():.4f}")

    # ── 5. Calibration ────────────────────────────────────────────────────────
    print("\n[5/5] Fitting calibration methods and evaluating ...")

    # Parametric — baselines (hard/voted labels)
    ts   = TemperatureScaling().fit(logits_cal, yh_cal_t)
    ps   = PlattScaling(N_CLASSES).fit(logits_cal, yh_cal_t)
    dc_h = DirichletCalibration(N_CLASSES).fit_hard(logits_cal, yh_cal_t)

    print(f"  T (TS):   {ts.T:.4f}")

    # Parametric — ours (soft labels)
    slts = SoftLabelTS().fit(logits_cal, ys_cal_t)
    mcts = MonteCarloTS(n_samples=50).fit(logits_cal, ys_cal_t)
    vs   = VectorScaling(N_CLASSES).fit(logits_cal, ys_cal_t)
    dc_s = DirichletCalibration(N_CLASSES).fit_soft(logits_cal, ys_cal_t)
    sp_s = SoftPlattScaling(N_CLASSES).fit(logits_cal, ys_cal_t)

    print(f"  T (SLTS): {slts.T:.4f}")
    print(f"  T (MCTS): {mcts.T:.4f}")

    # Non-parametric — baseline (hard labels)
    hb_hard = HardHistogramBinning(n_bins=n_bins).fit(probs_cal, yh_cal)

    # Non-parametric — ours (soft labels)
    hb = SoftHistogramBinning(n_bins=n_bins).fit(probs_cal, ys_cal)
    ir = SoftIsotonicRegression().fit(probs_cal, ys_cal)

    # ── Evaluation ────────────────────────────────────────────────────────────

    # Parametric: get full probability vectors
    p_ts   = apply_parametric(ts,   logits_all[idx_te])
    p_ps   = apply_parametric(ps,   logits_all[idx_te])
    p_dc_h = apply_parametric(dc_h, logits_all[idx_te])
    p_slts = apply_parametric(slts, logits_all[idx_te])
    p_mcts = apply_parametric(mcts, logits_all[idx_te])
    p_vs   = apply_parametric(vs,   logits_all[idx_te])
    p_dc_s = apply_parametric(dc_s, logits_all[idx_te])
    p_sp_s = apply_parametric(sp_s, logits_all[idx_te])

    main_results = []
    for name, p in [
        ("Uncalibrated",          probs_te),
        ("TS",                    p_ts),
        ("Platt (PS)",            p_ps),
        ("Dirichlet-Hard",        p_dc_h),
        ("MCTS (ours)",           p_mcts),
        ("SLTS (ours)",           p_slts),
        ("SoftPlatt (ours)",      p_sp_s),
        ("VS (ours)",             p_vs),
        ("Dirichlet-Soft (ours)", p_dc_s),
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
        pseudo_probs = np.zeros_like(probs_te)
        for i in range(len(probs_te)):
            pseudo_probs[i, pred[i]] = cal_conf[i]
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
        ("Uncalibrated", probs_te),
        ("TS", p_ts), ("Platt (PS)", p_ps),
        ("SLTS (ours)", p_slts), ("MCTS (ours)", p_mcts),
    ]:
        quant_results[name] = stratified_ece_soft(p, ys_te, n_bins=n_bins, n_quantiles=4)

    # Save all results
    output = {
        "dataset":          f"ChaosNLI-{subset}",
        "arch":             arch,
        "model_name":       config["hf_name"],
        "main_results":     main_results,
        "strat_results":    strat_results,
        "quant_results":    quant_results,
        "ts_temperature":   ts.T,
        "slts_temperature": slts.T,
        "mcts_temperature": mcts.T,
        "voted_label_accuracy": acc,
        "n_total":          N,
        "n_cal":            int(len(idx_cal)),
        "n_test":           int(len(idx_te)),
        "n_bins":           n_bins,
        "seed":             seed,
    }
    out_path = Path(results_dir) / f"chaosnli_{subset}_results_{arch}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ChaosNLI calibration experiment")
    p.add_argument("--cache-dir",   default="./cache",   help="directory for data and logit cache")
    p.add_argument("--results-dir", default="./results", help="directory for output JSON")
    p.add_argument("--n-bins",      type=int, default=15, help="ECE bins (default 15)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--arch",        default="roberta_large",
                   choices=["roberta_large", "deberta_v3"],
                   help="NLI model architecture (default: roberta_large)")
    p.add_argument("--subset",      default="combined",
                   choices=["snli", "mnli", "combined"],
                   help="ChaosNLI subset (default: combined = SNLI + MNLI)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {args.device}")
    run_experiment(args)
