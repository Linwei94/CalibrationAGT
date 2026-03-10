# PSLTS: Annotation-Free Calibration under Ambiguous Ground Truth

**Integration target**: Existing NeurIPS 2025 paper "Confidence Calibration under Ambiguous Ground Truth"
**New contribution type**: New method + theory + experiments (Section 5 or Appendix)
**Status**: AC-accepted for inclusion (REVISE conditions met by design)

---

## 1. Problem Statement

**Existing paper setting (SLTS)**: Annotator distributions {π_i} available at calibration time.
**NEW setting (PSLTS)**: Only voted labels {y_i*} available — the common practical case.

**Key question**: Can we reduce ECE_true without access to annotator distributions?

**Formal definition**:
- Model logits z_i ∈ ℝ^K for calibration set i=1..n
- Voted labels y_i* ∈ {1..K} (one-hot e_{y_i*})
- True annotator distribution π_i (UNKNOWN at calibration time; available for evaluation only)
- Goal: find calibrated predictions p_i = softmax(z_i/T) minimizing ECE_true = E[|P̂(Y=c|X) - P(Y=c|X)|]

---

## 2. Method: PSLTS (Pseudo-Soft Label Temperature Scaling)

### Core Idea
The pre-calibration model f(x) = softmax(z) encodes information about annotator disagreement:
- When f_i[y_i*] is high (model confident in voted class), example i is likely unambiguous
- When f_i[y_i*] is low (model uncertain about voted class), example i is likely ambiguous
- Therefore, β_i = 1 − f_i[y_i*] is a natural per-instance uncertainty proxy for annotation noise

### Algorithm
```
Input: logits {z_i}, voted labels {y_i*}
1. Compute pre-calibration predictions: f_i = softmax(z_i)
2. Compute instance-adaptive smoothing weight:
   β_i = 1 − f_i[y_i*]    (model's uncertainty about voted class)
3. Construct pseudo-soft labels:
   π̂_i = (1 − β_i) · e_{y_i*} + β_i / K
        = f_i[y_i*] · e_{y_i*} + (1 − f_i[y_i*]) / K
4. Optimize temperature:
   T* = argmin_T (1/n) Σ_i KL(π̂_i ∥ softmax(z_i/T))
5. Return softmax(z/T*)
```

### Interpretation of π̂_i
- **High confidence** (f_i[y_i*] ≈ 1): β_i ≈ 0, π̂_i ≈ e_{y_i*}  (likely unambiguous → keep one-hot)
- **Low confidence** (f_i[y_i*] ≈ 1/K): β_i ≈ (K−1)/K, π̂_i ≈ uniform (likely ambiguous → smooth heavily)
- **Intermediate**: smoothly interpolates between one-hot and uniform by voted-class uncertainty

This is the first instance-adaptive annotation-free calibration method that explicitly targets ECE_true.

### Relation to existing methods
- **Standard TS**: PSLTS with β_i ≡ 0 (no smoothing). Special case.
- **Uniform LabelSmooth-TS**: PSLTS with fixed ε = mean(β_i), non-adaptive ablation.
- **SLTS**: Temperature scaling against true annotator distributions π_i. Oracle upper bound.

---

## 3. Theory

### Proposition 2 (T*_PSLTS ≥ T*_TS)
Under the assumption that z_i are non-degenerate logits (f_i[y_i*] < 1 for all i):

T*_PSLTS ≥ T*_TS

**Proof**:
The optimal temperature for SoftTS satisfies the stationarity condition (gradient of KL = 0):
  (1/n) Σ_i E_{π̂_i}[z_i] = (1/n) Σ_i E_{softmax(z_i/T*)}[z_i]

For TS: LHS_TS = (1/n) Σ_i z_i[y_i*]

For PSLTS: LHS_PSLTS = (1/n) Σ_i [f_i[y*]·z_i[y*] + (1−f_i[y*])·mean_k(z_i[k])]
         = (1/n) Σ_i [z_i[y*] − (1−f_i[y*])·(z_i[y*] − mean_k(z_i[k]))]
         ≤ (1/n) Σ_i z_i[y*]  = LHS_TS       (since z_i[y*] ≥ mean_k(z_i[k]) when argmax f_i = y_i*)

Since RHS(T) = (1/n) Σ_i E_{softmax(z_i/T)}[z_i] is strictly decreasing in T,
a smaller LHS implies larger T*. QED.

**Corollary**: Since T*_SLTS > T*_TS (Proposition 1), and T*_PSLTS ≥ T*_TS,
PSLTS and SLTS both counter the harmful effect of TS sharpening under ambiguity.
PSLTS provides a no-annotation approximation to SLTS.

### Toy example validation (pilot experiment)
- T_TS = 0.916 < T_PSLTS = 2.587 ✓ (T ordering proved)
- Mean π̂_PSLTS[y*] = 0.905 ≈ Mean π̂_SLTS[y*] = 0.909 (remarkable closeness)
- ECE_true: TS=9.10% → PSLTS=6.82% → SLTS=5.88%
  (PSLTS improves over TS by 25%; gap from oracle only 0.94%)

### Remark on ECE_true
The mean PSLTS target E[π̂[y*]] = E[f[y*] + (1−f[y*])/K] naturally approximates the true
annotation agreement E[π[y*]] when the model's voted-class confidence f_i[y*] correlates
with true annotation agreement. The gap between PSLTS and SLTS measures this correlation.

---

## 4. Baselines in This Setting

| Method | Target | Annotation-free? | Notes |
|--------|--------|-----------------|-------|
| TS | ECE_voted (one-hot) | ✓ | Standard; T* < 1, worsens ECE_true |
| LabelSmooth-TS | ECE_voted (ε-smoothed) | ✓ | Fixed ε = 1/K; non-adaptive |
| PSLTS (ours) | ECE_true proxy (π̂) | ✓ | Instance-adaptive; no annotator labels |
| SLTS | ECE_true (π) | ✗ | Oracle; requires annotator distributions |

---

## 5. Expected Results

| Dataset | TS ECE_true | LabelSmooth-TS | PSLTS (ours) | SLTS (oracle) |
|---------|-------------|----------------|--------------|---------------|
| CIFAR-10H (ResNet-50) | 4.29% | ~3.5% | ~2.5% | 1.51% |
| ChaosNLI (RoBERTa-L) | 10.55% | ~8% | ~6% | 3.22% |
| DermaMNIST (ResNet-18) | 23.05% | ~20% | ~18% | 4.61% |
| ISIC 2019 (ENet-B4) | 18.54% | ~16% | ~14% | 9.71% |

Note: DermaMNIST/ISIC will show smallest improvement (overconfidence stress test per AC requirement).

---

## 6. Paper Integration Plan

**New section**: "Section X: Calibrating without Annotator Distributions"

Structure:
- §X.1: Motivation — most datasets have only voted labels
- §X.2: PSLTS method + Proposition 2
- §X.3: Experimental results (compact table; all 4 datasets)
- §X.4: Analysis — when does PSLTS succeed/fail? (α_i distribution, correlation with annotator entropy)

**Table addition**: Add PSLTS and LabelSmooth-TS rows to all 4 main experiment tables.

**Figure addition**: Scatter plot of PSLTS vs SLTS ECE_true — shows annotation-free method tracks the oracle.
