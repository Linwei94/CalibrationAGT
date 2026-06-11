"""
REAL open-vocabulary LLM calibration under ambiguous ground truth — AmbigNQ.

Dataset : sonny-dev/ambignq  (dev.json, 2002 examples).  Each question carries a
          SET of acceptable answer strings (union over singleAnswer.answer and
          every multipleQAs.qaPairs[i].answer).  Genuine ambiguity = >=2 distinct
          acceptable answer strings (the multipleQAs disambiguations).
Model   : Qwen/Qwen2.5-1.5B-Instruct on cuda, open-vocabulary SAMPLING.

Per question we sample S=10 free-form generations (do_sample, temp 1.0,
max_new_tokens=16).  Each generation is normalized (lowercase, strip punctuation
and leading articles) and clustered into semantic-equivalence classes by exact
normalized-string match -> a meaning distribution over clusters.

    confidence = top-cluster mass  (fraction of the S samples in the modal cluster).

The ambiguous ground truth is the SET of acceptable answers.  We reduce to K=2
[top-meaning, other]:
    p_acc = fraction of acceptable answers whose normalized form equals the
            model's top meaning  (p_acc in [0,1]).
    logits = [log conf, log(1-conf)]
    soft   = [p_acc, 1-p_acc]
    voted  = 0 if p_acc>=0.5 else 1
"""

import os, sys, json, re, string
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/root/CalibrationAGT/experiments")
from calibration import (TemperatureScaling, SoftLabelTS, MonteCarloTS,
                         LabelSmoothTS, apply_parametric)
from metrics import compute_all_metrics, annotation_entropy, print_results_table
from analyze_depth import save_bundle

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_CLASSES = 2
CAP = 1500
S = 10                       # generations per question
MAX_NEW_TOKENS = 16
TEMP = 1.0
SEED = 42
BUNDLE_PATH = "/tmp/real4/bundles/ambignq_llm_real.npz"
RESULTS_PATH = "/tmp/real4/ambignq_llm_real_results.json"

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text):
    """lowercase, strip punctuation, drop leading articles, collapse whitespace."""
    t = text.lower().strip()
    t = t.translate(_PUNCT)
    t = re.sub(r"\s+", " ", t).strip()
    toks = [w for w in t.split() if w not in _ARTICLES]
    return " ".join(toks)


def acceptable(ex):
    s = set()
    for a in ex["annotations"]:
        if a["type"] == "singleAnswer":
            s |= set(a["answer"])
        else:
            for qa in a["qaPairs"]:
                s |= set(qa["answer"])
    return ex["question"], s


def load_ambignq():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("sonny-dev/ambignq", "dev.json", repo_type="dataset")
    data = json.load(open(p))
    items = []
    for ex in data:
        q, acc = acceptable(ex)
        if len(acc) >= 2:                      # genuine ambiguity criterion
            items.append({"question": q, "acc": acc})
    return items


def load_llm(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
    return model, tok, dev


def build_prompt(tok, question):
    msgs = [{"role": "system",
             "content": "You are a helpful assistant. Answer the question with the "
                        "shortest possible answer (a few words), no explanation."},
            {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def sample_generations(model, tok, dev, prompt, n_samples):
    enc = tok(prompt, return_tensors="pt").to(dev)
    plen = enc.input_ids.shape[1]
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=TEMP,
        top_p=1.0,
        top_k=0,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=n_samples,
        pad_token_id=tok.pad_token_id,
    )
    gens = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
    return gens


def meaning_distribution(gens):
    """Cluster normalized generations by exact match -> (clusters dict, top_norm, conf)."""
    clusters = {}
    for g in gens:
        ng = normalize(g)
        if not ng:
            ng = "<empty>"
        clusters[ng] = clusters.get(ng, 0) + 1
    top_norm = max(clusters, key=clusters.get)
    conf = clusters[top_norm] / sum(clusters.values())
    return clusters, top_norm, conf


def stratified_split(voted, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(voted)
    cal_mask = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    for c in range(N_CLASSES):
        c_idx = idx[voted == c]
        rng.shuffle(c_idx)
        cal_mask[c_idx[: len(c_idx) // 2]] = True
    return idx[cal_mask], idx[~cal_mask]


def main():
    items = load_ambignq()
    print(f"AmbigNQ dev questions with >=2 distinct acceptable answers: {len(items)}")
    if len(items) > CAP:
        rng = np.random.default_rng(SEED)
        sel = sorted(rng.choice(len(items), CAP, replace=False))
        items = [items[i] for i in sel]
        print(f"  capped to {len(items)} for speed")

    mean_acc = float(np.mean([len(it["acc"]) for it in items]))
    print(f"mean #acceptable answers per question: {mean_acc:.3f}")

    model, tok, dev = load_llm(MODEL_NAME)
    print(f"loaded {MODEL_NAME} on {dev}")

    eps = 1e-6
    logits = np.zeros((len(items), N_CLASSES), dtype=np.float32)
    soft = np.zeros((len(items), N_CLASSES), dtype=np.float32)
    voted = np.zeros(len(items), dtype=np.int64)
    p_acc_list = np.zeros(len(items), dtype=np.float32)
    n_distinct_meanings = np.zeros(len(items), dtype=np.int64)
    examples_log = []

    for i, it in enumerate(items):
        prompt = build_prompt(tok, it["question"])
        gens = sample_generations(model, tok, dev, prompt, S)
        clusters, top_norm, conf = meaning_distribution(gens)
        n_distinct_meanings[i] = len(clusters)

        acc_norm = {normalize(a) for a in it["acc"]}
        acc_norm = {a for a in acc_norm if a}   # drop empties
        # p_acc = fraction of acceptable answers whose normalized form == top meaning.
        if len(acc_norm) > 0:
            p_acc = sum(1 for a in acc_norm if a == top_norm) / len(acc_norm)
        else:
            p_acc = 0.0
        p_acc_list[i] = p_acc

        conf_c = float(np.clip(conf, eps, 1 - eps))
        logits[i] = [np.log(conf_c), np.log(1 - conf_c)]
        soft[i] = [p_acc, 1 - p_acc]
        voted[i] = 0 if p_acc >= 0.5 else 1

        if i < 6:
            examples_log.append({
                "question": it["question"],
                "acc_norm": sorted(acc_norm)[:8],
                "top_meaning": top_norm, "conf": round(conf, 3),
                "n_meanings": len(clusters), "p_acc": round(p_acc, 3),
            })
        if (i + 1) % 200 == 0:
            print(f"  generated {i+1}/{len(items)}")

    # Diagnostics on the open-vocabulary distribution.
    frac_match = float((p_acc_list > 0).mean())
    mean_meanings = float(n_distinct_meanings.mean())
    print(f"\nopen-vocab diagnostics:")
    print(f"  mean distinct sampled meanings per question (S={S}): {mean_meanings:.3f}")
    print(f"  fraction of questions whose top meaning matches >=1 acceptable answer "
          f"(p_acc>0): {frac_match*100:.2f}%")
    print(f"  mean p_acc (soft top-meaning mass): {float(p_acc_list.mean()):.4f}")
    print(f"  voted-class balance: class0(top-meaning-correct)="
          f"{int((voted==0).sum())}  class1(other)={int((voted==1).sum())}")

    cal, te = stratified_split(voted)
    lc = torch.tensor(logits[cal], dtype=torch.float32)
    lt = logits[te]
    yh = torch.tensor(voted[cal], dtype=torch.long)
    ys = torch.tensor(soft[cal], dtype=torch.float32)
    probs_te = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()
    voted_te, soft_te = voted[te], soft[te]

    save_bundle(BUNDLE_PATH, name="ambignq_llm_real", n_classes=N_CLASSES,
                logits_cal=logits[cal], logits_te=logits[te],
                soft_cal=soft[cal], soft_te=soft[te],
                hard_cal=voted[cal], hard_te=voted[te])

    # Fit calibrators on the cal split.
    ts = TemperatureScaling().fit(lc, yh)
    slts = SoftLabelTS().fit(lc, ys)
    mcts1 = MonteCarloTS(n_samples=1).fit(lc, ys)
    lsts = LabelSmoothTS().fit(lc, yh)

    methods = [("Uncalibrated", probs_te),
               ("TS (voted)", apply_parametric(ts, lt)),
               ("SLTS", apply_parametric(slts, lt)),
               ("MCTS S=1", apply_parametric(mcts1, lt)),
               ("LS-TS", apply_parametric(lsts, lt))]

    res = []
    for nm, p in methods:
        r = compute_all_metrics(p, voted_te, soft_te, n_bins=15, name=nm)
        r["temperature"] = {"TS (voted)": ts.T, "SLTS": slts.T,
                            "MCTS S=1": mcts1.T, "LS-TS": lsts.T}.get(nm)
        res.append(r)

    print(f"\n[AmbigNQ open-vocab LLM]  K={N_CLASSES}  n_cal={len(cal)}  "
          f"n_test={len(te)}  mean H_test={annotation_entropy(soft_te).mean():.3f}")
    print_results_table(res)

    print("\n--- ECE_true / Brier / NLL (test split) ---")
    print(f"{'method':<14} {'ECE_true':>9} {'Brier_true':>11} {'NLL_true':>9} {'T':>7}")
    for r in res:
        T = r.get("temperature")
        print(f"{r['name']:<14} {r['ece_soft']:>9.4f} {r['brier_soft']:>11.4f} "
              f"{r['nll_soft']:>9.4f} {('' if T is None else f'{T:.3f}'):>7}")

    out = {
        "dataset": "AmbigNQ (sonny-dev/ambignq dev, SET-of-acceptable-answers ground truth)",
        "model": MODEL_NAME, "S": S, "max_new_tokens": MAX_NEW_TOKENS, "temp": TEMP,
        "n_questions": len(items), "mean_acceptable_answers": mean_acc,
        "mean_distinct_sampled_meanings": mean_meanings,
        "frac_top_meaning_in_acceptable_set": frac_match,
        "mean_p_acc": float(p_acc_list.mean()),
        "voted_class0": int((voted == 0).sum()), "voted_class1": int((voted == 1).sum()),
        "n_cal": len(cal), "n_test": len(te),
        "T_ts": ts.T, "T_slts": slts.T, "T_lsts": lsts.T,
        "main_results": res, "examples": examples_log,
        "bundle_path": BUNDLE_PATH,
    }
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nsaved -> {RESULTS_PATH}")
    print("bundle exists:", Path(BUNDLE_PATH).exists(), BUNDLE_PATH)


if __name__ == "__main__":
    main()
