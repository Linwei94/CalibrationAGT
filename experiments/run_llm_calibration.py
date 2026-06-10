"""
Calibration under ambiguous ground truth for LLMs / generative models
(TPAMI resubmission — Associate Editor point #3, Reviewer 2 point #3).

The paper's framework calibrates confidence against a *distribution over labels*
rather than a single voted label. This script shows the framework extends, with no
change to the calibrators, from fixed-K classifier logits to LLMs, in two regimes:

  (1) FIXED LABEL SET (e.g. multiple-choice QA, NLI):
      The LLM induces a categorical distribution over a known option set via the
      next-token probabilities of the option strings. When the task has a *human
      label distribution* (e.g. ChaosNLI's 100 annotators), this is exactly the
      paper's setting with LLM-derived logits -> TS/SLTS/MCTS/LS-TS apply verbatim.

  (2) OPEN VOCABULARY / SEQUENCES:
      For free-form generation there is no fixed K. We sample S generations, cluster
      them into semantic-equivalence classes (a meaning distribution), and take the
      model's confidence as the top-cluster mass. The "ambiguous ground truth" is the
      set of acceptable reference answers; the soft target is the probability that a
      randomly chosen acceptable answer matches the model's top meaning. This reduces
      to a 2-class top-meaning reliability problem to which the same calibrators apply.
      Calibrating to a single reference (voted) over-/under-states confidence exactly
      as in the classification case.

USAGE
-----
  python run_llm_calibration.py --demo                       # synthetic; verifies both regimes, no model
  python run_llm_calibration.py --mode labelset --model Qwen/Qwen2.5-0.5B-Instruct \
        --data prepared/nli_items.json                       # real LLM, fixed label set
  python run_llm_calibration.py --mode openvocab --model <hf-id> --data prepared/qa_items.json

DATA FORMATS (json list)
  labelset : [{"prompt": str, "options": [str,...], "pi": [float,...]}]   # pi = human label dist
  openvocab: [{"prompt": str, "references": [str,...]}]                    # acceptable answers
"""

import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibration import (TemperatureScaling, SoftLabelTS, MonteCarloTS, LabelSmoothTS,
                         DirichletCalibration, SoftIsotonicRegression, apply_parametric)
from metrics import compute_all_metrics, annotation_entropy, print_results_table


# ──────────────────────────────────────────────────────────────────────────────
# Shared calibrate + evaluate (logits + soft target + voted label -> metrics)
# Reuses the paper's calibrators unchanged.
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_and_report(logits, soft, voted, K, name, seed=42, n_bins=15):
    rng = np.random.default_rng(seed)
    n = len(voted); idx = rng.permutation(n); h = n // 2
    cal, te = idx[:h], idx[h:]
    lc = torch.tensor(logits[cal], dtype=torch.float32)
    lt = logits[te]
    yh = torch.tensor(voted[cal], dtype=torch.long)
    ys = torch.tensor(soft[cal], dtype=torch.float32)
    probs_cal = torch.softmax(lc, 1).numpy()
    probs_te  = torch.softmax(torch.tensor(lt, dtype=torch.float32), 1).numpy()
    voted_te, soft_te = voted[te], soft[te]

    ts = TemperatureScaling().fit(lc, yh)
    slts = SoftLabelTS().fit(lc, ys)
    mcts1 = MonteCarloTS(n_samples=1).fit(lc, ys)
    lsts = LabelSmoothTS().fit(lc, yh)
    methods = [("Uncalibrated", probs_te), ("TS (voted)", apply_parametric(ts, lt)),
               ("LS-TS", apply_parametric(lsts, lt)), ("SLTS", apply_parametric(slts, lt)),
               ("MCTS S=1", apply_parametric(mcts1, lt))]
    if K >= 3:   # Dirichlet needs >=2 classes; meaningful for K>=3
        dcs = DirichletCalibration(K).fit_soft(lc, ys)
        methods.append(("Dirichlet-Soft", apply_parametric(dcs, lt)))

    res = []
    for nm, p in methods:
        r = compute_all_metrics(p, voted_te, soft_te, n_bins=n_bins, name=nm)
        r["temperature"] = {"TS (voted)": ts.T, "SLTS": slts.T, "MCTS S=1": mcts1.T,
                            "LS-TS": lsts.T}.get(nm)
        res.append(r)
    print(f"\n[{name}]  K={K}  n_cal={len(cal)}  n_test={len(te)}  "
          f"mean H_test={annotation_entropy(soft_te).mean():.3f}")
    print_results_table(res)
    return {"name": name, "n_classes": K, "main_results": res,
            "T_ts": ts.T, "T_slts": slts.T, "T_lsts": lsts.T}


# ──────────────────────────────────────────────────────────────────────────────
# LLM probes
# ──────────────────────────────────────────────────────────────────────────────

def load_llm(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
    return model, tok, dev

def option_logits(model, tok, dev, prompt, options):
    """LLM-induced logits over a fixed option set: the total (teacher-forced) log-prob
    of each option string given the prompt, summed over the option's tokens. Scoring
    the full sequence (not just the first token) avoids collisions between options that
    share a first token or span multiple tokens."""
    base = tok(prompt, return_tensors="pt").input_ids.to(dev)
    vals = []
    with torch.no_grad():
        for o in options:
            ot = tok(" " + o, add_special_tokens=False).input_ids
            seq = torch.cat([base, torch.tensor([ot], device=dev)], dim=1)
            lp = torch.log_softmax(model(seq).logits[0].float(), -1)
            s = 0.0
            for j, tid in enumerate(ot):                  # log p(token_j | prompt, token_<j)
                s += float(lp[base.shape[1] + j - 1, tid])
            vals.append(s)
    return np.array(vals, dtype=np.float32)               # used as logits (softmax => distribution)

def _norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\b(a|an|the)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

def sample_meanings(model, tok, dev, prompt, S=10, max_new=16):
    gens = []
    ids = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        for _ in range(S):
            out = model.generate(**ids, do_sample=True, temperature=1.0, top_p=0.95,
                                  max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
            gens.append(tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True))
    clusters = {}
    for g in gens:
        clusters.setdefault(_norm(g), 0)
        clusters[_norm(g)] += 1
    return clusters  # meaning -> count


# ──────────────────────────────────────────────────────────────────────────────
# Regime (1): fixed label set  (NLI / MCQA with human label distribution)
# ──────────────────────────────────────────────────────────────────────────────

def run_labelset(items, model=None, tok=None, dev=None):
    K = len(items[0]["options"])
    logits, soft = [], []
    for it in items:
        soft.append(np.asarray(it["pi"], np.float32))
        if model is not None:
            logits.append(option_logits(model, tok, dev, it["prompt"], it["options"]))
        else:                       # demo: synthetic over-confident, NON-separable logits
            logits.append(np.asarray(it["_demo_logits"], np.float32))
    logits = np.stack(logits); soft = np.stack(soft)
    voted = soft.argmax(1).astype(np.int64)
    return calibrate_and_report(logits, soft, voted, K, "LLM label-set (NLI/MCQA)")


# ──────────────────────────────────────────────────────────────────────────────
# Regime (2): open vocabulary  (sequence generation -> 2-class top-meaning reliability)
# ──────────────────────────────────────────────────────────────────────────────

def run_openvocab(items, model=None, tok=None, dev=None, S=10):
    logits, soft, voted = [], [], []
    for it in items:
        refs = [_norm(r) for r in it["references"]]
        if model is not None:
            clusters = sample_meanings(model, tok, dev, it["prompt"], S=S)
        else:                       # demo: synthetic meaning distribution
            clusters = it["_demo_clusters"]
        total = sum(clusters.values())
        top = max(clusters, key=clusters.get)
        conf = clusters[top] / total                       # model's top-meaning mass
        # ambiguous ground truth: fraction of acceptable references equal to the top meaning
        p_acc = float(np.mean([r == top for r in refs])) if refs else 0.0
        conf = min(max(conf, 1e-3), 1 - 1e-3)
        logits.append([np.log(conf), np.log(1 - conf)])    # 2-class top-meaning logits
        soft.append([p_acc, 1 - p_acc])                    # soft target over {top, other}
        voted.append(0 if p_acc >= 0.5 else 1)             # single-reference (voted) label
    logits = np.asarray(logits, np.float32); soft = np.asarray(soft, np.float32)
    voted = np.asarray(voted, np.int64)
    return calibrate_and_report(logits, soft, voted, 2, "LLM open-vocab (semantic clusters)")


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic demo data (verifies both regimes without a model or external data)
# ──────────────────────────────────────────────────────────────────────────────

def demo_labelset(seed=0, n=400, K=3):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        a = rng.dirichlet(np.ones(K) * rng.choice([0.4, 2.0]))   # mix sharp + ambiguous
        # synthetic over-confident model: sharpen log-pi and add noise so the logits are
        # NOT a monotone function of pi (avoids a perfectly separable voted-label problem).
        z = (np.log(a + 1e-6) + rng.normal(0, 0.8, K)) / 0.55
        items.append({"prompt": "demo", "options": [f"c{k}" for k in range(K)],
                      "pi": a.tolist(), "_demo_logits": z.tolist()})
    return items

def demo_openvocab(seed=0, n=400, R=5):
    """Each item has a latent acceptability a_i: the probability that the model's top
    meaning is an acceptable answer. We draw R reference answers (ambiguous ground
    truth) each equal to the top meaning w.p. a_i, giving a SOFT target p_acc in (0,1).
    The model's top-cluster mass (confidence) is correlated with a_i but over-confident,
    so calibrating to the single/majority reference (voted) differs from calibrating to
    the soft acceptability -- the paper's contrast, in the generative setting."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        a = float(rng.beta(2.0, 2.0))                         # latent acceptability in (0,1)
        refs = ["ans top" if rng.random() < a else f"ans alt{rng.integers(3)}" for _ in range(R)]
        # over-confident model: top-cluster mass exceeds true acceptability
        conf = min(0.97, a + 0.25 * (1 - a))
        alt = 1 - conf
        clusters = {"ans top": int(round(conf * 20)) + 1, "ans other": int(round(alt * 20)) + 1}
        items.append({"prompt": "demo", "references": refs, "_demo_clusters": clusters})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["labelset", "openvocab"], default="labelset")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model", help="HF causal-LM id (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument("--data", help="json items (see header)")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--samples", type=int, default=10, help="generations per item (openvocab)")
    args = ap.parse_args()

    model = tok = dev = None
    if args.model:
        model, tok, dev = load_llm(args.model)
        print(f"loaded {args.model} on {dev}")

    if args.demo:
        items = demo_labelset() if args.mode == "labelset" else demo_openvocab()
    elif args.data:
        items = json.load(open(args.data))
    else:
        ap.error("specify --demo or --data")

    out = (run_labelset(items, model, tok, dev) if args.mode == "labelset"
           else run_openvocab(items, model, tok, dev, S=args.samples))
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    p = Path(args.results_dir) / f"llm_{args.mode}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
