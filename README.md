# Confidence Calibration under Ambiguous Ground Truth

> **NeurIPS 2026 submission** — Anonymous authors

Standard post-hoc calibrators (Temperature Scaling, Platt scaling, Histogram Binning) assume a unique ground-truth label per input. When multiple annotators genuinely disagree—as in medical imaging, NLI, and perceptual tasks—this assumption produces a systematic **calibration gap**: the calibrator appears well-calibrated under majority-vote evaluation while remaining significantly overconfident against the true annotator distribution.

We formalise the gap, prove it is a structural consequence of the voted-label target (not a capacity limitation), and propose six soft-label post-hoc calibrators that close it—requiring no retraining.

## Key Results

### Motivating example (3-class toy, controlled setting)

| Method | ECE-voted ↓ | ECE-true ↓ | Gap Δ |
|---|---|---|---|
| Uncalibrated | 2.8% | 10.4% | +7.6 pp |
| Temperature Scaling | **1.1%** | 12.2% | +11.1 pp |
| Platt Scaling | 1.4% | 12.2% | +10.8 pp |
| HB-Hard | 2.5% | 12.1% | +9.5 pp |
| **SLTS (ours)** | 14.9% | **3.9%** | −11.0 pp |

*All hard-label baselines reduce ECE-voted but widen the true-label gap. SLTS uses the soft annotator distribution as target and closes it.*

### Main experiments — ECE-true (%) across all benchmarks

| Method | C10H R50 | C10H ViT | NLI Rob | NLI Deb | ISIC ENet | ISIC ViT | Derm R18 | Derm ViT |
|---|---|---|---|---|---|---|---|---|
| Uncalibrated | 4.97 | 4.99 | 27.79 | 35.97 | 25.76 | 23.31 | 34.35 | 37.12 |
| TS | 4.29 | 4.48 | 10.55 | 11.63 | 18.54 | 16.59 | 23.05 | 24.36 |
| Dir.-Hard | 4.46 | 4.70 | 11.55 | 12.69 | 20.36 | 17.76 | 24.58 | 24.84 |
| SLTS (ours) | 1.51 | 0.85 | 3.22 | 3.45 | 9.71 | 6.96 | 4.61 | 3.58 |
| Dir.-Soft (ours) | 1.25 | **0.72** | **2.57** | 3.19 | 7.50 | 6.85 | 3.55 | **3.32** |
| IR-Soft (ours) | **0.72** | 0.91 | 2.65 | **2.15** | **1.75** | **1.73** | **2.15** | 3.79 |

*Dir.-Hard (most expressive hard-label method) fails to improve over TS → the bottleneck is the voted-label target, not capacity. Dirichlet-Soft achieves the best Brier score and NLL across all benchmarks.*

## Methods

Six post-hoc calibrators, all operating on cached logits — no retraining needed:

| Method | Params | Key idea |
|---|---|---|
| **SLTS** | 1 scalar $T$ | Minimise KL(π̂ ∥ softmax(z/T)); same family as TS, soft target |
| **MCTS** | 1 scalar $T$ | Draw S annotation samples per example; converges to SLTS as S→∞ |
| **VS** | K scalars | Per-class temperatures for class-asymmetric disagreement |
| **IR-Soft** | non-param | Monotone confidence mapping via PAVA with soft targets |
| **SoftPlatt** | 2K params | Diagonal affine W·z+b with soft KL target |
| **Dirichlet-Soft** | K²+K params | Full K×K affine with soft KL target + ODIR regularisation |

## Repository Structure

```
experiments/
  calibration.py        # All six calibration methods
  metrics.py            # ECE-true, aECE, cwECE, Brier, NLL
  run_cifar10h.py       # CIFAR-10H (ResNet-50, ViT-B/16)
  run_chaosnli.py       # ChaosNLI (RoBERTa-Large, DeBERTa-v3)
  run_isic2019.py       # ISIC 2019 (EfficientNet-B4, ViT-S/16)
  run_dermamnist.py     # DermaMNIST (ResNet-18, ViT-S/16)

toy_example/
  run_toy_example.py    # Controlled 3-class motivating experiment

paper/
  main.tex              # NeurIPS 2026 paper (7-page main body + appendix)
  make_figures.py       # Reproduce all figures
  references.bib
  neurips_2025.sty

results/                # JSON outputs from experiment runs
```

## Reproducing the Paper

### Dependencies

```bash
pip install torch torchvision transformers numpy matplotlib scikit-learn scipy tqdm
```

### 1. Toy experiment (Section 4)

```bash
python toy_example/run_toy_example.py
```

### 2. CIFAR-10H (Section 7.1)

```bash
python experiments/run_cifar10h.py --arch resnet50 --device cuda
python experiments/run_cifar10h.py --arch vit_b16  --device cuda
```

CIFAR-10 downloads automatically; CIFAR-10H annotations are fetched from the [official repo](https://github.com/jcpeterson/cifar-10h).

### 3. ChaosNLI (Section 7.2)

Download the dataset from the [ChaosNLI Dropbox](https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip) and extract to `experiments/cache/chaosNLI_v1.0/`.

```bash
python experiments/run_chaosnli.py --arch roberta_large --device cuda
python experiments/run_chaosnli.py --arch deberta_v3    --device cuda
```

### 4. ISIC 2019 (Section 7.3)

Download from [ISIC Archive 2019](https://challenge.isic-archive.com/data/#2019) and place under `experiments/data/isic2019/`.

```bash
python experiments/run_isic2019.py --arch efficientnet_b4 --device cuda
python experiments/run_isic2019.py --arch vit_s16         --device cuda
```

### 5. DermaMNIST (Appendix)

```bash
python experiments/run_dermamnist.py --arch resnet18 --device cuda
python experiments/run_dermamnist.py --arch vit_s16  --device cuda
```

### 6. Figures and paper

```bash
cd paper && python make_figures.py
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Reference

This work is the calibration analogue of:

> D. Stutz, K. Dvijotham, A. T. Cemgil, A. Doucet.
> *Conformal Prediction under Ambiguous Ground Truth.*
> TMLR, 2023. [arXiv:2307.09302](https://arxiv.org/abs/2307.09302)

The ISIC 2019 annotator model mirrors the reader study in:

> Y. Liu et al. *A deep learning system for differential diagnosis of skin diseases.*
> Nature Medicine, 26:900–908, 2020.
