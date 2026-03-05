# Confidence Calibration under Ambiguous Ground Truth

> **NeurIPS 2025 submission** — Anonymous authors

Standard post-hoc calibration methods (Temperature Scaling, Platt scaling, histogram binning) assume a unique ground-truth label per input. When labels are *ambiguous*—where rational annotators genuinely disagree—these methods incur a systematic **calibration gap**: they appear well-calibrated under hard-label evaluation while remaining significantly overconfident relative to the true annotator distribution. This paper formalises the gap, proves it is a structural consequence of the majority-vote target (not a capacity limitation), and proposes simple soft-label post-hoc calibrators that close it.

## Key idea

| Method | ECE-Hard ↓ | ECE-Soft ↓ | Gap Δ |
|---|---|---|---|
| Uncalibrated | 3.3% | 10.6% | +7.3 pp |
| Temperature Scaling | **3.2%** | 10.3% | +7.1 pp |
| Platt Scaling | 3.0% | 10.1% | +7.1 pp |
| HB-Hard | 3.3% | 10.6% | +7.3 pp |
| **SLTS (ours)** | 9.9% | **3.9%** | −6.0 pp |

*All hard-label baselines leave ECE-Soft ≈ 10–11% unchanged. SLTS replaces the voted-label target with the annotator distribution and reduces ECE-Soft by 62%.*

## Methods

- **SLTS** — Soft-Label Temperature Scaling: minimises KL(π̂ ∥ softmax(z/T))
- **MCTS** — Monte Carlo Temperature Scaling: samples individual annotations; converges to SLTS as S→∞
- **VS** — Vector Scaling with soft labels: per-class temperatures
- **HB-Soft / IR-Soft** — non-parametric calibration with soft targets

## Repository structure

```
paper/
  main.tex            # NeurIPS 2025 paper
  make_figures.py     # Reproduces all figures (run from paper/)
  references.bib
  neurips_2025.sty

experiments/
  calibration.py      # Calibration methods (SLTS, MCTS, VS, HB-Soft, IR-Soft)
  metrics.py          # ECE-Hard, ECE-Soft, Brier, NLL
  run_cifar10h.py     # CIFAR-10H experiment (ResNet-50)
  run_isic2019.py     # ISIC 2019 experiment (EfficientNet-B4)
  run_dermamnist.py   # DermaMNIST experiment (ResNet-18, appendix)
  run_mcts_analysis.py

toy_example/
  run_toy_example.py  # Controlled 3-class motivating experiment
```

## Reproducing the paper

### 1. Motivating toy experiment
```bash
python toy_example/run_toy_example.py
```

### 2. CIFAR-10H (Section 7.1)
```bash
python experiments/run_cifar10h.py --device cuda --epochs 30
```
Downloads CIFAR-10 automatically; CIFAR-10H annotations are fetched from the [official repo](https://github.com/jcpeterson/cifar-10h).

### 3. ISIC 2019 (Section 7.2)
Download the dataset from [ISIC Archive 2019](https://challenge.isic-archive.com/data/#2019) and place it as:
```
data/
  ISIC_2019_Training_Input/   # ~25k dermoscopic images
  ISIC_2019_Training_GroundTruth.csv
```
Then run:
```bash
python experiments/run_isic2019.py --device cuda --epochs 20 \
    --data_dir data/ISIC_2019_Training_Input \
    --gt_csv   data/ISIC_2019_Training_GroundTruth.csv
```

### 4. Generate all figures
```bash
cd paper && python make_figures.py
```
Results JSONs from steps 2–3 are loaded automatically if present; otherwise schematic placeholder values are used.

### 5. Compile the paper
```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn scipy tqdm
```

## Reference

This work is the calibration analogue of:

> D. Stutz, K. Dvijotham, A. T. Cemgil, and A. Doucet.
> *Conformal Prediction under Ambiguous Ground Truth.*
> TMLR, 2023.

The ISIC 2019 experiment mirrors the skin-disease reader study in:

> Y. Liu et al. *A deep learning system for differential diagnosis of skin diseases.*
> Nature Medicine, 26:900–908, 2020.
