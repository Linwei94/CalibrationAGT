# Confidence Calibration under Ambiguous Ground Truth

> **Submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**
>
> Linwei Tao, Haoyang Luo, Minjing Dong, Chang Xu

## Overview

Standard post-hoc calibration assumes a unique ground-truth label per input and fits calibrators against majority-voted labels.
We show this practice is fundamentally flawed under label ambiguity: Temperature Scaling is biased toward temperatures that underestimate annotator uncertainty, with the true-label miscalibration gap growing monotonically with annotation entropy.

We propose a family of ambiguity-aware post-hoc calibrators that optimise proper scoring rules against the full label distribution and require no model retraining, spanning three practical annotation regimes:

| Regime | Method | Description |
|--------|--------|-------------|
| Full annotator distribution available | Dirichlet-Soft, SoftPlatt, VS, IR-Soft | Optimise soft-label objectives against the annotator distribution |
| Individual annotations (not aggregated) | MCTS *S*=1 | Single randomly drawn annotation per example suffices |
| Voted labels only | LS-TS | Data-driven pseudo-soft target from model confidence |

## Main Results

`ECE_true` (%, lower is better) across 8 settings:

| Method | C10H R50 | C10H ViT | NLI RoB | NLI DeB | ISIC ENet | ISIC ViT | Derm R18 | Derm ViT |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|
| TS | 4.29 | 4.48 | 10.55 | 11.63 | 18.72 | 17.04 | 22.25 | 24.65 |
| LS-TS | 1.57 | 2.37 | 4.16 | 2.65 | 9.34 | 15.49 | 5.05 | 7.36 |
| MCTS *S*=1 | 1.45 | 0.81 | 2.82 | 3.20 | 10.09 | 7.47 | 5.07 | 3.28 |
| SLTS | 1.51 | 0.85 | 3.22 | 3.45 | 9.76 | 7.10 | 5.13 | 3.29 |
| Dirichlet-Soft | **1.25** | **0.72** | **2.57** | 3.19 | 8.35 | 6.32 | 3.94 | 3.18 |
| IR-Soft | 0.72 | 0.91 | 2.65 | **2.15** | **1.77** | **2.05** | **2.20** | **2.06** |

## Repository Structure

```
experiments/
  calibration.py            # All calibration methods (TS, SLTS, MCTS, LS-TS, VS, IR-Soft, SoftPlatt, Dirichlet-Soft, ...)
  metrics.py                # ECE_true, aECE, cwECE, Brier, NLL
  run_toy_example.py        # Toy motivating example (Figure 1)
  run_cifar10h.py           # CIFAR-10H experiments
  run_chaosnli.py           # ChaosNLI experiments
  run_isic2019.py           # ISIC 2019 experiments
  run_dermamnist.py         # DermaMNIST experiments
  run_multiseed.py          # Multi-seed stability ablation
  run_cal_size_ablation.py  # Calibration set size ablation
  run_lsts_ablation.py      # LS-TS smoothing strategy ablation
  run_ats_comparison.py     # ATS vs LS-TS comparison
  add_mcts_s1.py            # Add MCTS S=1 results to existing JSON files
  add_oracle_ts.py          # Add Oracle TS upper bound to existing JSON files
  plot_entropy_validation.py     # Figure 3: entropy vs calibration error
  plot_reliability_diagrams.py   # Reliability diagram figures
  results/                  # JSON files with all experimental results
```

## Requirements

```bash
pip install torch torchvision transformers numpy matplotlib scikit-learn scipy tqdm pillow
```

Tested with Python 3.12, PyTorch 2.x.

## Data Preparation

Download the following datasets and place them under `experiments/cache/`:

- **CIFAR-10H**: [CIFAR-10H repository](https://github.com/jcpeterson/cifar-10h) — `cifar10h-probs.npy` + CIFAR-10 test logits
- **ChaosNLI**: [ChaosNLI repository](https://github.com/easonnie/ChaosNLI) — extract `chaosNLI_v1.0/` under `cache/`
- **ISIC 2019**: [ISIC Archive](https://challenge.isic-archive.com/data/) — training CSV + images
- **DermaMNIST**: downloaded automatically via `medmnist` package

Pre-extracted logits (model outputs on val/test sets) are provided in `cache/` to allow reproducing calibration results without re-running model training.

Model checkpoints are available at: https://huggingface.co/linweitao/calibration-agt-checkpoints
Download and place `.pt`/`.pth` files under `experiments/cache/` to reproduce training results.

## Reproducing Results

**Toy example:**
```bash
python experiments/run_toy_example.py
```

**Main experiments:**
```bash
python experiments/run_cifar10h.py --arch resnet50 --device cuda
python experiments/run_cifar10h.py --arch vit_b16 --device cuda

python experiments/run_chaosnli.py --arch roberta_large --device cuda
python experiments/run_chaosnli.py --arch deberta_v3 --device cuda

python experiments/run_isic2019.py --arch efficientnet_b4 --device cuda
python experiments/run_isic2019.py --arch vit_s16 --device cuda

python experiments/run_dermamnist.py --arch resnet18 --device cuda
python experiments/run_dermamnist.py --arch vit_s16 --device cuda
```

**Ablations:**
```bash
python experiments/run_multiseed.py
python experiments/run_cal_size_ablation.py
python experiments/run_lsts_ablation.py
python experiments/run_ats_comparison.py
```

**Figures:**
```bash
python experiments/plot_entropy_validation.py --figures-dir paper/figs
python experiments/plot_reliability_diagrams.py
```

## Results Files

All results are stored as JSON files under `experiments/results/`. Each file contains a `main_results` list with per-method metrics (`ece_sampled`, `brier_sampled`, `nll_sampled`, `temperature`, etc.).
