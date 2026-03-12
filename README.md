# Confidence Calibration under Ambiguous Ground Truth

> **NeurIPS 2026 submission** — Anonymous authors

This repository studies post-hoc calibration when an input admits multiple plausible labels. The paper distinguishes:

- `ECE_voted`: calibration against the majority-voted one-hot label
- `ECE_true`: calibration against labels drawn from the underlying ambiguous label distribution reflected by annotator disagreement

The main finding is that standard post-hoc calibrators such as Temperature Scaling, Platt scaling, and Histogram Binning can improve `ECE_voted` while still leaving large `ECE_true`. Our methods address both practical regimes:

- `Ambiguity-aware calibrators` when annotator distributions are available at calibration time: `SLTS`, `MCTS`, `VS`, `IR-Soft`, `SoftPlatt`, `Dirichlet-Soft`
- `LS-TS` when the calibration set contains only one-hot voted labels

## Main results

### Toy motivating example

The controlled 3-class toy example uses three Gaussian clusters:

- class 0: `N((-3.2, 1.1), diag(0.60, 0.45))`, `pi=[1,0,0]`
- ambiguous middle cluster: `N((0, 0), diag(1.15, 0.75))`, `pi=[0,0.70,0.30]`
- class 2: `N((3.2, -1.1), diag(0.60, 0.45))`, `pi=[0,0,1]`

The middle cluster is always majority-voted as class 1, so standard calibrators see it as a one-hot target even though the underlying label distribution is `70/30`.

| Method | ECE_voted ↓ | ECE_true ↓ |
|---|---:|---:|
| Uncalibrated | 1.34 | 7.54 |
| TS | 1.26 | 8.88 |
| Platt (PS) | **0.77** | 8.70 |
| HB-Hard | 1.28 | 8.79 |

This is the qualitative point used in Figure 1 and Table 1 of the paper: traditional post-hoc methods help `ECE_voted`, but they do not fix `ECE_true`.

### Benchmarks

Main paper benchmarks:

- `CIFAR-10H`: repeated human labels expose perceptual ambiguity
- `ChaosNLI`: 100 human labels expose semantic ambiguity
- `ISIC 2019`: synthetic dermatologist readers model clinically plausible differential diagnosis
- `DermaMNIST`: supplementary medical benchmark in the appendix

Representative `ECE_true` results from Table 2:

| Method | C10H R50 | C10H ViT | NLI Rob | NLI Deb | ISIC ENet | ISIC ViT | Derm R18 | Derm ViT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TS | 4.29 | 4.48 | 10.55 | 11.63 | 18.54 | 16.59 | 23.05 | 24.36 |
| LS-TS | 1.57 | 2.37 | 4.16 | 2.65 | 9.34 | 15.49 | 5.05 | 7.36 |
| SLTS | 1.51 | 0.85 | 3.22 | 3.45 | 9.71 | 6.96 | 4.61 | 3.58 |
| Dir.-Soft | 1.25 | **0.72** | **2.57** | 3.19 | 7.50 | 6.85 | 3.55 | **3.32** |
| IR-Soft | **0.72** | 0.91 | 2.65 | **2.15** | **1.75** | **1.73** | **2.15** | 3.79 |

## Repository structure

```text
experiments/
  calibration.py
  metrics.py
  run_toy_example.py
  plot_data_distribution.py
  run_cifar10h.py
  run_chaosnli.py
  run_isic2019.py
  run_dermamnist.py

paper/
  main.tex
  main.pdf
  make_figures.py
  make_intro_ambiguity_figure.py
  make_fig4_stratified.py
  figs/

results/
```

## Reproducing the paper

Dependencies:

```bash
pip install torch torchvision transformers numpy matplotlib scikit-learn scipy tqdm
```

Toy figure:

```bash
python experiments/run_toy_example.py
```

CIFAR-10H:

```bash
python experiments/run_cifar10h.py --arch resnet50 --device cuda
python experiments/run_cifar10h.py --arch vit_b16 --device cuda
```

ChaosNLI:

```bash
python experiments/run_chaosnli.py --arch roberta_large --device cuda
python experiments/run_chaosnli.py --arch deberta_v3 --device cuda
```

ISIC 2019:

```bash
python experiments/run_isic2019.py --arch efficientnet_b4 --device cuda
python experiments/run_isic2019.py --arch vit_s16 --device cuda
```

DermaMNIST:

```bash
python experiments/run_dermamnist.py --arch resnet18 --device cuda
python experiments/run_dermamnist.py --arch vit_s16 --device cuda
```

Paper:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Notes

- `paper/main.tex` is the authoritative manuscript source.
- `proposal.md` is an earlier planning document and intentionally preserves some historical framing that predates the current paper draft.
