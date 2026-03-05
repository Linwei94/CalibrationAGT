# Confidence Calibration under Ambiguous Ground Truth

**Draft Proposal** — March 2026

---

## 1. Motivation

A well-calibrated classifier should have the property that when it predicts a class with confidence $p$, that class should appear with frequency $p$ among predictions made with that confidence. Formally, for a classifier outputting a probability vector $\hat{p}(x) \in \Delta^K$:

$$P(\hat{Y} = Y \mid \hat{p}(\hat{Y}) = p) = p \quad \forall p \in [0,1]$$

This definition—and the entire calibration literature—rests on a silent assumption: **there exists a single, unambiguous ground truth label** $Y$ for every input $x$.

In practice, this assumption frequently fails. In medical imaging, a skin lesion may be legitimately classified as either benign or malignant by different dermatologists. In natural language inference, whether a premise "entails" a hypothesis is often genuinely debatable. In autonomous driving, whether an object is a "pedestrian" or a "cyclist" at low resolution is inherently ambiguous. In all such settings, the ground truth is not a single label but a **distribution over labels**, reflecting the irreducible uncertainty of the task.

The standard response is to aggregate annotations—taking a majority vote or a single annotator's label—and treat the result as ground truth. Calibration methods such as Temperature Scaling (TS) [Guo et al., 2017], Platt Scaling, and histogram binning are then applied against these aggregated hard labels. We argue this leads to a systematic failure:

> **Standard calibration methods, when trained against hard (voted) labels, produce models that are calibrated with respect to the vote—but not with respect to the true underlying label distribution.**

This is the direct analog of the coverage gap identified by Stutz et al. [2023] for conformal prediction: just as standard CP underestimates uncertainty when calibrated against voted labels, standard calibration methods produce overconfident models for inherently ambiguous inputs.

---

## 2. Related Work

### 2.1 Confidence Calibration

- **Temperature Scaling (TS)** [Guo et al., 2017]: Post-hoc calibration by learning a single scalar temperature $T$ to rescale logits. Minimizes NLL on a calibration set. Simple and effective but assumes a unique ground truth.
- **Histogram Binning / Isotonic Regression** [Zadrozny & Elkan, 2001, 2002]: Non-parametric calibration methods that map model confidences to empirical frequencies.
- **Label Smoothing** [Szegedy et al., 2016]: Implicitly encodes label uncertainty during training by distributing a small fraction of probability mass to non-target classes. Not a calibration method per se and uses a fixed uniform distribution.
- **Soft calibration** [Brier, 1950]: Proper scoring rules (Brier score, log loss) with soft targets have been studied in the meteorology and forecasting literature.

### 2.2 Calibration under Distribution Shift

- **CPCS** [Park et al., 2020], **TransCal** [Wang et al., 2020]: Calibration under covariate shift. Different from our setting where the label distribution itself is uncertain.

### 2.3 Learning from Multiple Annotators

- **Dawid-Skene** [1979]: EM algorithm to infer true labels from multiple noisy annotators.
- **CrowdLayer** [Rodrigues & Pereira, 2018]: End-to-end learning from multiple annotators.
- **Union set annotation** [Peterson et al., 2019]: Studies the case where annotators provide a set of acceptable labels.
- **Soft labels from annotator disagreement** [Uma et al., 2021]: Studies calibration and learning with soft labels derived from annotations.

### 2.4 Conformal Prediction under Ambiguous Ground Truth

- **Stutz et al. [2023]**: Proposes Monte Carlo conformal prediction (MCCP), which samples calibration labels from the annotator distribution rather than using a single voted label. Achieves coverage guarantees with respect to the true label distribution. Reveals a "coverage gap" for standard CP.

### 2.5 Gap in the Literature

Despite the large body of work on calibration and on learning from multiple annotators, **the interaction between label ambiguity and confidence calibration has not been systematically studied**. Specifically:
- No existing work formally defines calibration with respect to a label distribution (as opposed to a single label).
- No existing work shows whether and why standard calibration methods fail under ambiguous ground truth.
- No principled post-hoc calibration method exists for the multi-annotator / ambiguous setting.

---

## 3. Problem Formulation

### 3.1 Setup

Let $\mathcal{X}$ be an input space and $\mathcal{Y} = \{1, \ldots, K\}$ be a label space. Instead of a fixed ground truth label $Y \in \mathcal{Y}$, we assume there exists an **annotator label distribution** $\pi(\cdot \mid x) \in \Delta^{\mathcal{Y}}$ for each input $x$, representing the probability that a randomly drawn annotator assigns each label. In practice, $\pi(\cdot \mid x)$ is estimated from $m$ annotator labels $\{a_1, \ldots, a_m\} \subset \mathcal{Y}$ via:

$$\hat{\pi}_k(x) = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}[a_i = k]$$

The **voted label** is $y^* = \arg\max_k \hat{\pi}_k(x)$ (or the single observed label when $m=1$).

A classifier $f : \mathcal{X} \to \Delta^{\mathcal{Y}}$ outputs a predicted probability vector $\hat{p}(x) = f(x)$.

### 3.2 Standard Calibration (Hard Label)

Standard calibration requires:

$$P(Y = c \mid \hat{p}_c(X) = p) = p \quad \forall c, p$$

where $Y$ is a single hard label (e.g., voted label $y^*$). This is what TS and other post-hoc methods optimize.

### 3.3 Calibration under Ambiguous Ground Truth

We propose the following definition of **soft calibration**:

$$\mathbb{E}[\pi_c(X) \mid \hat{p}_c(X) = p] = p \quad \forall c, p$$

That is, the model's predicted probability for class $c$ should equal the expected annotator probability for class $c$, conditioned on the model's prediction. This reduces to standard calibration when $\pi(\cdot \mid x)$ is always a one-hot distribution.

**Equivalently**, for the top-predicted class $\hat{c}(x) = \arg\max_k \hat{p}_k(x)$:

$$\mathbb{E}\left[\pi_{\hat{c}}(X) \mid \max_k \hat{p}_k(X) = p\right] = p$$

The gap between this expectation and $p$ defines the **soft calibration error (SCE)**:

$$\mathrm{SCE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \overline{\pi}_{\hat{c}}(B_b) - \overline{p}(B_b) \right|$$

where $B_b$ is the $b$-th confidence bin, $\overline{p}(B_b)$ is the mean confidence in the bin, and $\overline{\pi}_{\hat{c}}(B_b)$ is the mean annotator probability for the predicted class.

### 3.4 The Calibration Gap

For ambiguous inputs (where $\pi(\cdot \mid x)$ is not one-hot), we expect:

$$\mathbb{E}[\pi_{\hat{c}}(X) \mid \max_k \hat{p}_k(X) = p] < \mathbb{E}[\mathbf{1}[Y^* = \hat{c}(X)] \mid \max_k \hat{p}_k(X) = p]$$

because the voted label $Y^*$ over-represents the majority class. This means a model calibrated on voted labels will **overestimate** the probability of the majority class for ambiguous inputs—a systematic overconfidence bias.

---

## 4. Why Standard Calibration Fails: Theoretical Intuition

### 4.1 The Voting Bias

Consider a binary classification setting ($K=2$). For an ambiguous input, suppose annotators assign label 1 with probability $\pi_1 = 0.6$ and label 2 with probability $\pi_2 = 0.4$. The voted label is $Y^* = 1$. A model trained on single labels drawn from $\pi$ learns $f_1(x) \approx 0.6$. Temperature Scaling optimizes NLL on the calibration set:

$$\mathcal{L}_{TS} = -\mathbb{E}[\log \hat{p}_{Y^*}(X)/T]$$

This objective treats $Y^*=1$ as always correct, pushing the calibrated confidence toward matching the frequency of voted-label-1 samples. For ambiguous inputs, the model learns to be confident in the majority class, **ignoring the true uncertainty of 0.4 in class 2**.

### 4.2 Temperature Scaling Cannot Capture Class-Conditional Ambiguity

Temperature Scaling applies a single scalar $T$ to rescale all logits. It cannot separately adjust confidence for ambiguous vs. unambiguous inputs—it can only globally compress or expand the probability simplex. In contrast, soft calibration requires class-conditional, input-dependent adjustments that a single temperature parameter is fundamentally unable to capture.

### 4.3 The ECE Measurement Illusion

If calibration is evaluated using hard labels (ECE-Hard), standard TS will appear well-calibrated because both the calibration objective and the evaluation metric use the same voted labels. The miscalibration only becomes visible when evaluating with soft labels (ECE-Soft), exposing the "calibration gap"—the direct analog of the coverage gap in conformal prediction.

---

## 5. Proposed Methods

### 5.1 Monte Carlo Temperature Scaling (MCTS)

**Inspired by** Monte Carlo conformal prediction [Stutz et al., 2023], we propose to calibrate by **sampling labels from the annotator distribution** rather than using voted labels.

Given $m$ annotator labels $\{a_1, \ldots, a_m\}$ for each calibration example $x_i$, instead of computing:

$$\mathcal{L}_{TS} = -\frac{1}{n} \sum_{i=1}^{n} \log \hat{p}_{y_i^*}(x_i) / T$$

we compute:

$$\mathcal{L}_{MCTS} = -\frac{1}{n \cdot m} \sum_{i=1}^{n} \sum_{j=1}^{m} \log \hat{p}_{a_{ij}}(x_i) / T$$

This is equivalent to calibrating TS against the empirical label distribution rather than the voted label, and can be implemented with a simple modification to the calibration loss.

**Theoretical guarantee**: Under mild conditions, MCTS optimizes a proper scoring rule with respect to the annotator label distribution, yielding unbiased calibration targets.

### 5.2 Soft-Label Temperature Scaling (SLTS)

If the annotator label distribution $\hat{\pi}(x)$ is known (as a soft label vector), we can directly minimize the KL divergence between the calibrated model output and the soft label:

$$\mathcal{L}_{SLTS} = \frac{1}{n} \sum_{i=1}^{n} \mathrm{KL}(\hat{\pi}(x_i) \| \mathrm{softmax}(z_i / T))$$

where $z_i$ are the model logits. This is equivalent to minimizing the cross-entropy with soft targets, which is a proper scoring rule and guarantees unbiased calibration with respect to $\hat{\pi}$.

### 5.3 Class-Wise Soft Calibration (CWSC)

Class-wise temperature scaling [Guo et al., 2017 variant] applies a separate temperature $T_k$ per class. Combined with soft labels:

$$\mathcal{L}_{CWSC} = \frac{1}{n} \sum_{i=1}^{n} \mathrm{KL}(\hat{\pi}(x_i) \| \mathrm{softmax}(z_i \odot T^{-1}))$$

This allows class-specific calibration adjustments, potentially better capturing class-conditional ambiguity patterns.

### 5.4 Ambiguity-Aware Binning Methods

Extend histogram binning and isotonic regression to use soft calibration targets: in each confidence bin, the calibration target is the mean annotator probability for the predicted class (rather than the hard-label accuracy). This is a natural extension that requires no assumption about the form of ambiguity.

---

## 6. Theoretical Analysis

### 6.1 Proper Scoring Rules

**Theorem (informal)**: The KL divergence $\mathrm{KL}(\hat{\pi}(x) \| f(x))$ is a proper scoring rule with respect to the annotator distribution $\pi(\cdot \mid x)$, meaning it is uniquely minimized when $f(x) = \hat{\pi}(x)$.

**Corollary**: Methods optimizing this loss (SLTS, CWSC) are guaranteed to produce soft-calibrated models as $n \to \infty$ and as model capacity increases.

### 6.2 Calibration Error Bounds

We aim to derive finite-sample bounds on the SCE for our proposed methods, analogous to the coverage guarantees in [Stutz et al., 2023]. Key challenge: unlike the distribution-free conformal guarantees, calibration bounds depend on model class and data distribution.

### 6.3 Relationship to Hard Calibration

**Claim**: Soft calibration with hard labels reduces to standard calibration. Soft calibration is strictly stronger: a soft-calibrated model is also hard-calibrated (with respect to any label sampled from $\pi$), but not vice versa.

---

## 7. Experimental Plan

### 7.1 Datasets with Annotator Disagreement

- **CIFAR-10H** [Peterson et al., 2019]: ~50 human annotations per image for the CIFAR-10 test set.
- **NIH Chest X-Ray** / **CheXpert**: Multiple radiologist labels for thoracic disease classification.
- **Skin Lesion Classification (HAM10000 / ISIC)**: Dermatologist annotations with inherent ambiguity (used in [Stutz et al., 2023]).
- **ChaosNLI** [Nie et al., 2020]: Collective human annotations for NLI with captured disagreement.

### 7.2 Baselines

- Uncalibrated model (raw softmax)
- Temperature Scaling (TS) with hard labels [Guo et al., 2017]
- Platt Scaling / Histogram Binning / Isotonic Regression with hard labels
- Label Smoothing (LS) [Szegedy et al., 2016]

### 7.3 Proposed Methods (Ours)

- Monte Carlo Temperature Scaling (MCTS)
- Soft-Label Temperature Scaling (SLTS)
- Class-Wise Soft Calibration (CWSC)
- Ambiguity-Aware Binning

### 7.4 Evaluation Metrics

**Primary (soft calibration):**
- Soft ECE (SCE): ECE computed against soft labels $\hat{\pi}(x)$
- Soft Brier Score: $\mathbb{E}[\|\hat{p}(X) - \hat{\pi}(X)\|_2^2]$
- Soft NLL: $-\mathbb{E}[\sum_k \hat{\pi}_k(X) \log \hat{p}_k(X)]$

**Secondary (hard calibration, for comparison):**
- Standard ECE against voted labels
- Hard Brier Score
- Accuracy

**Stratified analysis:**
- Calibration metrics split by ambiguity level (e.g., entropy of $\hat{\pi}(x)$): unambiguous vs. moderately ambiguous vs. highly ambiguous.

### 7.5 Key Hypotheses to Verify

1. **H1 (Calibration Gap)**: Standard TS exhibits a significant gap between ECE-Hard and ECE-Soft, especially for ambiguous samples.
2. **H2 (Soft Calibration)**: SLTS and MCTS reduce ECE-Soft significantly compared to TS.
3. **H3 (No Hard Calibration Regression)**: Soft-calibrated models maintain competitive ECE-Hard.
4. **H4 (Ambiguity-Stratified)**: The gap between TS and our methods is largest for high-ambiguity samples.

---

## 8. Motivating Toy Example

To establish the existence and magnitude of the calibration gap, we construct a controlled synthetic experiment:

**Setup**: A 3-class classification problem with:
- Class 0: "clear" inputs, always labeled 0 ($\pi_0 = 1.0$)
- Class 1: "ambiguous" inputs, annotated as class 1 with probability 0.6 and class 2 with probability 0.4
- Class 2: "clear" inputs, always labeled 2 ($\pi_2 = 1.0$)

**Features**: 2D Gaussian clusters, one per class center.

**Procedure**:
1. Train a 2-layer MLP on single-draw hard labels.
2. Apply Temperature Scaling calibrated on hard labels.
3. Evaluate calibration using both ECE-Hard and ECE-Soft.
4. Plot reliability diagrams for both metrics.

**Concrete findings** (from `toy_example/run_toy_example.py`):

| Method       | ECE-Hard | ECE-Soft | T (temperature) |
|---|---|---|---|
| Uncalibrated | 0.027    | 0.078    | — |
| TS           | **0.017**| **0.092**| 0.865 < 1 ← wrong direction! |
| SLTS (ours)  | 0.149    | 0.078    | 2.660 > 1 ← correct direction |

Stratified ECE-Soft by input ambiguity:

| Method | Ambiguous inputs | Clear inputs |
|---|---|---|
| TS            | **0.248** | 0.024 |
| SLTS (ours)   | **0.050** | 0.127 |

Key insight: TS calibrated on voted labels finds T < 1 (decreases temperature → MORE confident), because for the ambiguous cluster all voted labels are always class 1 so the model appears "underconfident" in the hard sense. This is precisely the wrong direction: soft calibration requires T > 1 to bring the model's confidence down to the true annotator probability of 0.70.

See `toy_example/run_toy_example.py` for the implementation.

---

## 9. Expected Contributions

1. **Formal definition** of confidence calibration under ambiguous ground truth (soft calibration), with a clear connection to existing proper scoring rule theory.
2. **Empirical demonstration** of the calibration gap: standard TS is miscalibrated with respect to annotator label distributions.
3. **Proposed methods**: Monte Carlo TS, Soft-Label TS, and ambiguity-aware binning—simple, practical, post-hoc calibration methods for the ambiguous setting.
4. **Theoretical analysis**: Proper scoring rule guarantees for proposed methods; finite-sample error bounds.
5. **Comprehensive experiments** on real-world datasets with annotator disagreement, establishing new benchmarks for soft calibration.

---

## 10. Discussion and Broader Impact

### 10.1 When Does This Matter?

The calibration gap will be small when label ambiguity is low (most medical AI systems with clear diagnoses) and large when label ambiguity is high (rare diseases, edge cases, low-resolution inputs). Our work is most impactful precisely in the high-stakes, high-ambiguity settings where calibration is most critical for decision-making.

### 10.2 Connections to Uncertainty Quantification

Soft calibration is closely related to epistemic vs. aleatoric uncertainty: the annotator label distribution $\hat{\pi}(x)$ captures aleatoric (irreducible) uncertainty, while model uncertainty captures epistemic (reducible) uncertainty. A soft-calibrated model correctly represents aleatoric uncertainty, which is a necessary condition for useful uncertainty quantification in safety-critical systems.

### 10.3 Limitations

- Requires access to multiple annotations at calibration time (not always available).
- Soft calibration metrics require knowledge of $\hat{\pi}(x)$, which is an approximation.
- Connection between MCTS theoretical guarantees and finite-sample behavior needs careful analysis.

---

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
- Stutz, D., et al. (2023). Conformal prediction under ambiguous ground truth. *TMLR*. [[arXiv:2307.09302]](https://arxiv.org/abs/2307.09302)
- Peterson, J. C., et al. (2019). Human uncertainty makes classification more robust. *ICCV 2019*.
- Uma, A., et al. (2021). Learning from disagreement: A survey. *JAIR 2021*.
- Nie, Y., et al. (2020). What can we learn from collective human opinions on natural language inference data? *EMNLP 2020*.
- Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. *Applied Statistics*.
- Zadrozny, B., & Elkan, C. (2001, 2002). Obtaining calibrated probability estimates; Transforming classifier scores into accurate multiclass probability estimates. *KDD*.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*.
