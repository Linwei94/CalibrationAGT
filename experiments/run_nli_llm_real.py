"""
REAL LLM calibration under ambiguous ground truth (TPAMI AE #3).

Dataset : ChaosNLI v1.0 (Nie et al., EMNLP 2020) -- 100 REAL human annotations
          per premise/hypothesis pair over {entailment, neutral, contradiction}.
          Mirrored on HF at lguerdan/indeterminacy-datasets (SNLI + MNLI-m subsets).
Model   : Qwen/Qwen2.5-1.5B-Instruct (fallback 0.5B), teacher-forced option log-prob.

Label index convention (canonical, matches stanfordnlp/snli ClassLabel features):
    0 = entailment, 1 = neutral, 2 = contradiction.
ChaosNLI label_counter keys 'e'/'n'/'c' are mapped into that order.
"""

import os, sys, json
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/root/CalibrationAGT/experiments")
from calibration import (TemperatureScaling, SoftLabelTS, MonteCarloTS, LabelSmoothTS,
                         DirichletCalibration, SoftIsotonicRegression, apply_parametric)
from metrics import compute_all_metrics, annotation_entropy, print_results_table
from analyze_depth import save_bundle

OPTIONS = ["entailment", "neutral", "contradiction"]   # index order = soft-label order
KEY = {"e": 0, "n": 1, "c": 2}
N_CLASSES = 3
CAP = 3000
SEED = 42


def load_chaosnli():
    from huggingface_hub import hf_hub_download
    items = []
    for fn in ["chaosNLI_v1.0/chaosNLI_snli.jsonl",
               "chaosNLI_v1.0/chaosNLI_mnli_m.jsonl"]:
        p = hf_hub_download("lguerdan/indeterminacy-datasets", fn, repo_type="dataset")
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                lc = d["label_counter"]
                counts = np.array([lc.get("e", 0), lc.get("n", 0), lc.get("c", 0)],
                                  dtype=np.float64)
                tot = counts.sum()
                if tot < 3:                      # require >=3 valid annotator labels
                    continue
                ex = d["example"]
                items.append({"premise": ex["premise"], "hypothesis": ex["hypothesis"],
                              "soft": (counts / tot).astype(np.float32)})
    return items


def build_prompt(p, h):
    return (f"Premise: {p}\nHypothesis: {h}\n"
            "Does the premise entail, contradict, or is neutral to the hypothesis? "
            "Answer with one word: entailment, neutral, or contradiction.\nAnswer:")


def load_llm(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
    return model, tok, dev


def option_logits(model, tok, dev, prompt, options):
    """Total teacher-forced log-prob of each option string (summed over its tokens)."""
    base = tok(prompt, return_tensors="pt").input_ids.to(dev)
    vals = []
    with torch.no_grad():
        for o in options:
            ot = tok(" " + o, add_special_tokens=False).input_ids
            seq = torch.cat([base, torch.tensor([ot], device=dev)], dim=1)
            lp = torch.log_softmax(model(seq).logits[0].float(), -1)
            s = 0.0
            for j, tid in enumerate(ot):
                s += float(lp[base.shape[1] + j - 1, tid])
            vals.append(s)
    return np.array(vals, dtype=np.float32)


def sanity_accuracy_standard_snli(model, tok, dev, n=400, seed=0):
    """Correctness gate: the SAME prompt + label-index convention must clear >0.55 on
    STANDARD SNLI validation. ChaosNLI is the adversarially-selected MAX-disagreement
    subset, so voted-label accuracy there is low BY CONSTRUCTION; this check isolates
    prompt/label-order bugs from genuine ground-truth ambiguity."""
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/snli", split="validation").filter(lambda x: x["label"] != -1)
    rng = np.random.default_rng(seed)
    sel = sorted(rng.choice(len(ds), min(n, len(ds)), replace=False))
    ds = ds.select(sel)
    gold = np.array(ds["label"])          # SNLI features: 0=ent,1=neu,2=con
    correct = 0
    for p, h, g in zip(ds["premise"], ds["hypothesis"], gold):
        if option_logits(model, tok, dev, build_prompt(p, h), OPTIONS).argmax() == g:
            correct += 1
    return correct / len(ds)


def stratified_split(voted, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(voted)
    cal_mask = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    for c in range(N_CLASSES):
        c_idx = idx[voted == c]
        rng.shuffle(c_idx)
        cal_mask[c_idx[:len(c_idx) // 2]] = True
    return idx[cal_mask], idx[~cal_mask]


def main():
    items = load_chaosnli()
    print(f"ChaosNLI (SNLI+MNLI-m) examples with >=3 annotators: {len(items)}")
    if len(items) > CAP:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(items), CAP, replace=False)
        items = [items[i] for i in sorted(sel)]
        print(f"  capped to {len(items)} for speed")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        model, tok, dev = load_llm(model_name)
    except Exception as e:
        print(f"  {model_name} failed ({repr(e)[:120]}); falling back to 0.5B")
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        model, tok, dev = load_llm(model_name)
    print(f"loaded {model_name} on {dev}")

    soft = np.stack([it["soft"] for it in items])
    voted = soft.argmax(1).astype(np.int64)

    logits = np.zeros((len(items), N_CLASSES), dtype=np.float32)
    for i, it in enumerate(items):
        logits[i] = option_logits(model, tok, dev, build_prompt(it["premise"], it["hypothesis"]), OPTIONS)
        if (i + 1) % 500 == 0:
            print(f"  scored {i+1}/{len(items)}")

    acc = float((logits.argmax(1) == voted).mean())
    print(f"\nVoted-label top-1 accuracy on ChaosNLI: {acc*100:.2f}%  "
          "(low BY CONSTRUCTION -- ChaosNLI is the max-disagreement subset)")

    # CORRECTNESS GATE: same prompt + label order on STANDARD SNLI must clear > 0.55.
    gate_acc = sanity_accuracy_standard_snli(model, tok, dev)
    print(f"CORRECTNESS GATE: standard-SNLI top-1 accuracy = {gate_acc*100:.2f}%")
    if gate_acc <= 0.55:
        print("GATE FAILED (<=0.55 on standard SNLI). Stopping -- likely a prompt/label-order bug.")
        sys.exit(2)
    print("GATE PASSED (> 0.55 on standard SNLI) -> prompt & label-index convention are correct.")

    cal, te = stratified_split(voted)
    lc = torch.tensor(logits[cal], dtype=torch.float32)
    lt = logits[te]
    yh = torch.tensor(voted[cal], dtype=torch.long)
    ys = torch.tensor(soft[cal], dtype=torch.float32)
    probs_cal = torch.softmax(lc, 1).numpy()
    probs_te = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()
    voted_te, soft_te = voted[te], soft[te]

    # Save bundle (matches the published split/logits)
    bundle_path = "/tmp/real3/bundles/nli_llm_real.npz"
    save_bundle(bundle_path, name="nli_llm_real", n_classes=N_CLASSES,
                logits_cal=logits[cal], logits_te=logits[te],
                soft_cal=soft[cal], soft_te=soft[te],
                hard_cal=voted[cal], hard_te=voted[te])

    # Fit calibrators on cal split
    ts = TemperatureScaling().fit(lc, yh)
    slts = SoftLabelTS().fit(lc, ys)
    mcts1 = MonteCarloTS(n_samples=1).fit(lc, ys)
    lsts = LabelSmoothTS().fit(lc, yh)
    dcs = DirichletCalibration(N_CLASSES).fit_soft(lc, ys)
    ir = SoftIsotonicRegression().fit(probs_cal, soft[cal])

    methods = [("Uncalibrated", probs_te),
               ("TS (voted)", apply_parametric(ts, lt)),
               ("SLTS", apply_parametric(slts, lt)),
               ("MCTS S=1", apply_parametric(mcts1, lt)),
               ("LS-TS", apply_parametric(lsts, lt)),
               ("Dirichlet-Soft", apply_parametric(dcs, lt))]
    # IR-Soft: monotone map on top-class confidence -> reconstruct prob matrix
    cal_conf, pred = ir.calibrate(probs_te)
    pir = np.zeros_like(probs_te)
    for i in range(len(probs_te)):
        pir[i, pred[i]] = cal_conf[i]
        rest = np.ones(N_CLASSES) * (1 - cal_conf[i]) / (N_CLASSES - 1)
        rest[pred[i]] = 0
        pir[i] += rest
    methods.append(("IR-Soft", pir))

    res = []
    for nm, p in methods:
        r = compute_all_metrics(p, voted_te, soft_te, n_bins=15, name=nm)
        r["temperature"] = {"TS (voted)": ts.T, "SLTS": slts.T, "MCTS S=1": mcts1.T,
                            "LS-TS": lsts.T}.get(nm)
        res.append(r)

    print(f"\n[ChaosNLI LLM label-set]  K={N_CLASSES}  n_cal={len(cal)}  n_test={len(te)}  "
          f"mean H_test={annotation_entropy(soft_te).mean():.3f}")
    print_results_table(res)

    out = {"dataset": "ChaosNLI (SNLI+MNLI-m, 100 human annotators/ex)",
           "model": model_name, "n_examples": len(items),
           "top1_accuracy": acc, "gate_accuracy_standard_snli": gate_acc,
           "n_cal": len(cal), "n_test": len(te),
           "main_results": res, "T_ts": ts.T, "T_slts": slts.T, "T_lsts": lsts.T}
    Path("/tmp/real3").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("/tmp/real3/nli_llm_real_results.json", "w"), indent=2)
    print("\nsaved -> /tmp/real3/nli_llm_real_results.json")
    print("bundle exists:", Path(bundle_path).exists(), bundle_path)


if __name__ == "__main__":
    main()
