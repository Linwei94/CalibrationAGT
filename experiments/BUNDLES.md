# Building depth-analysis bundles

`analyze_depth.py` (Reviewer-1 W3/R3 diagnostics + theory A2/A3 numbers) runs on a
small `.npz` "bundle" per (dataset, architecture). The most faithful way to create a
bundle is to dump the exact calibration/test arrays each `run_*.py` already computes,
so the bundle matches the published split, logits, and synthetic-annotation seed.

## Quick check (no cache needed)

```bash
/path/to/python analyze_depth.py --demo      # synthetic data; verifies the four diagnostics
```

## Real data: add 4 lines to each run script

Pull the logit cache first (`cache/` from the HF checkpoints repo). Then add, in each
`run_*.py`, immediately **after the calibration/test arrays are materialised and
before fitting** (i.e. just above the `ts = TemperatureScaling().fit(...)` line):

**`run_cifar10h.py` and `run_chaosnli.py`** (cal/test arrays already named the same):
```python
from analyze_depth import save_bundle
save_bundle(f"{results_dir}/bundles/{Path(out_path).stem if False else arch}.npz",
            name=f"{arch}", n_classes=N_CLASSES,
            logits_cal=logits_cal, logits_te=logits_te,
            soft_cal=ys_cal, soft_te=ys_te, hard_cal=yh_cal, hard_te=yh_te)
```
(simplest: `name=f"cifar10h_{arch}"` / `f"chaosnli_{arch}"`, path `results/bundles/<name>.npz`.)

**`run_isic2019.py` and `run_dermamnist.py`** (calibrate on val, evaluate on test):
```python
from analyze_depth import save_bundle
save_bundle(f"{args.results_dir}/bundles/{DATASET}_{arch}.npz",
            name=f"{DATASET}_{arch}", n_classes=N_CLASSES,
            logits_cal=logits_val, logits_te=logits_test,
            soft_cal=ys_val,       soft_te=ys_test,
            hard_cal=val_labels,   hard_te=test_labels)
```
with `DATASET = "isic2019"` / `"dermamnist"`.

`save_bundle` accepts numpy arrays or torch tensors, so passing either the
`logits_cal` tensor (CIFAR/NLI) or the `logits_val` numpy array (ISIC/Derm) works.

## Run the analysis

```bash
python run_cifar10h.py  --arch resnet50          # now also writes results/bundles/cifar10h_resnet50.npz
...                                               # repeat for all 8 (dataset, arch) runs
python analyze_depth.py --bundle-dir results/bundles    # analyse all; writes results/depth/*.json
```

## Outputs (one JSON per bundle, under `results/depth/`)

- **D1 `D1_entropy_bins`** — ECE_true per annotation-entropy bin, per method.
  Use to show gains concentrate in high-entropy examples (Reviewer 1 R3).
  Tie-robust: handles the class-conditional medical data (many identical entropies).
- **D2 `D2_confidence_bins`** — population reliability gap `mean(conf) − mean(π_pred)`
  per confidence bin, per method. Use to show which confidence bins each method fixes.
- **D3 `D3_class_structure`** — per-(voted)class ECE_true and `eta2_entropy_by_class`
  (η² of annotation entropy explained by class). High η² (ISIC/Derm, class-conditional)
  ⇒ class-aware methods (VS/Dirichlet-Soft) should help; low η² (CIFAR-10H, instance-level)
  ⇒ they are unnecessary. Directly answers "are class-aware methods needed?"
- **D4 `D4_lsts_proxy`** — `v̄−ū` (signed proxy gap; >0 ⇒ over-smoothing) and
  `ρ=corr(u,v)`, with `T_SLTS`, `T_LS-TS`. Fills Table `tab:lsts-valid` (Appendix A3)
  and confirms Proposition "LS-TS proxy validity".
