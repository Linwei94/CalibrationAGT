# Experiment Progress

**Last updated:** 2026-03-19
**Current phase:** Phase 8–9 (Paper Writing + Polish)
**Idea round:** 1 (accepted)
**Paper target:** IEEE TPAMI

## Key Results Summary

All main experiments, ablations, and robustness analyses are complete. The central finding is confirmed: the calibration **target** (voted-label vs soft-label), not method capacity, is the limiting factor under label ambiguity. Dirichlet-Soft achieves the best Brier/NLL in 7/8 settings. MCTS S=1 (single annotation per example) matches the full-distribution SLTS baseline within 0.6 pp ECE across all 8 settings — the key practical finding. LS-TS reduces ECE by 7–78% without any annotator data. Paper draft is complete (27 pages) and undergoing final polishing.

## Main Experiment Results

All values: ECE = ECE_true (%), Br = Brier score (soft), NLL = NLL (soft).

### CIFAR-10H (real annotations, ~51 per image, K=10)

| Method | T (R50) | ECE (R50) | Br (R50) | NLL (R50) | T (ViT) | ECE (ViT) | Br (ViT) | NLL (ViT) |
|--------|---------|-----------|----------|-----------|---------|-----------|----------|-----------|
| TS | 2.03 | 4.29 | 0.112 | 0.363 | 2.04 | 4.48 | 0.106 | 0.346 |
| LS-TS | 3.08 | 1.57 | 0.109 | 0.293 | 2.76 | 2.37 | 0.103 | 0.285 |
| MCTS S=1 | 3.14 | 1.45 | 0.111 | 0.296 | 3.07 | 0.81 | 0.101 | 0.277 |
| SLTS | 3.18 | 1.51 | 0.110 | 0.293 | 3.07 | 0.85 | 0.102 | 0.278 |
| Dir-Soft | --- | 1.25 | 0.109 | 0.271 | --- | 0.72 | 0.100 | 0.255 |
| IR-Soft | --- | 0.72 | 0.115 | 0.340 | --- | 0.91 | 0.105 | 0.321 |

### ChaosNLI (real annotations, 100 per example, K=3)

| Method | T (RoB) | ECE (RoB) | Br (RoB) | NLL (RoB) | T (DeB) | ECE (DeB) | Br (DeB) | NLL (DeB) |
|--------|---------|-----------|----------|-----------|---------|-----------|----------|-----------|
| TS | 2.42 | 10.55 | 0.537 | 0.886 | 3.81 | 11.63 | 0.549 | 0.900 |
| LS-TS | 4.40 | 4.16 | 0.522 | 0.873 | 6.15 | 2.65 | 0.529 | 0.881 |
| MCTS S=1 | 3.49 | 2.82 | 0.519 | 0.862 | 5.45 | 3.20 | 0.529 | 0.877 |
| SLTS | 3.41 | 3.22 | 0.520 | 0.862 | 5.28 | 3.45 | 0.529 | 0.877 |
| Dir-Soft | --- | 2.57 | 0.510 | 0.839 | --- | 3.19 | 0.509 | 0.836 |
| IR-Soft | --- | 2.65 | 0.549 | 0.934 | --- | 2.15 | 0.554 | 0.942 |

### ISIC 2019 (synthetic annotations, m=9, K=8)

| Method | T (ENet) | ECE (ENet) | Br (ENet) | NLL (ENet) | T (ViT) | ECE (ViT) | Br (ViT) | NLL (ViT) |
|--------|----------|------------|-----------|------------|---------|-----------|----------|-----------|
| TS | 1.77 | 18.72 | 0.559 | 1.662 | 1.23 | 17.04 | 0.630 | 1.578 |
| LS-TS | 4.30 | 9.34 | 0.528 | 1.149 | 3.98 | 15.49 | 0.619 | 1.303 |
| MCTS S=1 | 4.51 | 10.09 | 0.530 | 1.149 | 2.77 | 7.47 | 0.595 | 1.248 |
| SLTS | 4.42 | 9.76 | 0.529 | 1.149 | 2.68 | 7.10 | 0.594 | 1.248 |
| Dir-Soft | --- | 8.35 | 0.518 | 1.086 | --- | 6.32 | 0.568 | 1.184 |
| IR-Soft | --- | 1.77 | 0.539 | 1.288 | --- | 2.05 | 0.627 | 1.479 |

### DermaMNIST (synthetic annotations, m=5, K=7)

| Method | T (R18) | ECE (R18) | Br (R18) | NLL (R18) | T (ViT) | ECE (ViT) | Br (ViT) | NLL (ViT) |
|--------|---------|-----------|----------|-----------|---------|-----------|----------|-----------|
| TS | 1.84 | 22.25 | 0.687 | 1.638 | 2.55 | 24.65 | 0.683 | 1.664 |
| LS-TS | 3.03 | 5.05 | 0.630 | 1.394 | 4.19 | 7.36 | 0.615 | 1.362 |
| MCTS S=1 | 3.42 | 5.07 | 0.629 | 1.388 | 5.04 | 3.28 | 0.609 | 1.347 |
| SLTS | 3.43 | 5.13 | 0.629 | 1.388 | 5.03 | 3.29 | 0.609 | 1.346 |
| Dir-Soft | --- | 3.94 | 0.612 | 1.305 | --- | 3.18 | 0.598 | 1.279 |
| IR-Soft | --- | 2.20 | 0.640 | 1.468 | --- | 2.06 | 0.626 | 1.423 |

## Ablation Results

| Experiment | Key Finding | Status |
|-----------|-------------|--------|
| LS-TS smoothing strategies (Table 5) | Data-driven global ε (LS-TS) matches per-class CC-LS; fixed ε=0.1 degrades below TS | completed |
| ATS vs LS-TS (Table 3) | ATS (adaptive T, voted target) fails to improve over TS in all 8 settings; LS-TS (global T, soft target) reduces ECE 7–78% | completed |
| MCTS convergence (Table 1) | S=1 matches S→∞ within 0.02 pp ECE on CIFAR-10H R50; T* variance drops with S | completed |
| Calibration set size (8 settings) | Soft methods stabilise at 5–10% of cal data; TS ECE flat regardless of cal set size | completed |
| Annotation count (6 settings) | Soft methods robust at m≥2; TS ECE *increases* with m (target mismatch widens) | completed |
| Multi-seed stability (8 settings) | Std devs 2–10× smaller than method gaps; rankings robust across seeds | completed |

## Analysis Results

| Analysis | Key Finding | Status |
|---------|-------------|--------|
| Entropy validation (Fig. 3) | TS pointwise error increases monotonically with annotation entropy H(x) on all 3 dataset-arch combos, confirming Proposition 2 | completed |
| Dirichlet-Hard vs TS | Dir-Hard (most expressive voted-label method) consistently *underperforms* TS — capacity ≠ target | completed |
| ISIC confusion matrix | Clinically calibrated 8×8 matrix; mean diagonal 75% matching Liu et al. 2020; MEL/NV most confused pair | completed |

## Timeline

| Date | Event | Outcome |
|------|-------|---------|
| 2026-02 | Pilot experiments (CIFAR-10H R50) | SLTS reduces ECE from 4.29% to 1.51%; concept validated |
| 2026-02 | Full experiments: 4 benchmarks × 2 architectures | All 8 settings confirm target > capacity finding |
| 2026-03 | Added Dirichlet-Soft, SoftPlatt, VS, IR-Soft | Dir-Soft best Brier/NLL in 7/8; IR-Soft best ECE on medical |
| 2026-03 | Added LS-TS (annotation-free) | Reduces ECE 7–78% with voted labels only |
| 2026-03 | Added ATS baseline | Confirms adaptive T with wrong target doesn't help |
| 2026-03 | All robustness ablations completed | Multi-seed, cal-size, annotation-count all stable |
| 2026-03-12 | Added MCTS S=1 to all 8 settings | Matches SLTS within 0.6 pp ECE everywhere — key finding |
| 2026-03-19 | Paper polish: TPAMI style, consistency fixes, SLTS de-emphasised | 27-page draft, zero compile warnings |

## Issues & Observations

- **ATS/LS-TS ablation tables use slightly different ECE computation for CIFAR-10H** (`ece` vs `ece_sampled`); footnoted in paper. Relative ordering unaffected.
- **LS-TS weak on ISIC ViT-S/16** (15.49% ECE, only 7% reduction): model confidence is a poor proxy for annotation ambiguity on this architecture-dataset pair.
- **MCTS S=1 occasionally outperforms SLTS** (e.g., ChaosNLI RoBERTa: 2.82% vs 3.22%): stochastic objective may provide implicit regularisation.
- **IR-Soft achieves lowest ECE on medical benchmarks** but worst Brier/NLL — non-parametric flexibility helps ECE but not distributional fit.
