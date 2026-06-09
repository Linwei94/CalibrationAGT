# Building depth-analysis bundles

`analyze_depth.py` (Reviewer-1 W3/R3 diagnostics + theory A2/A3 numbers) runs on a
small `.npz` "bundle" per (dataset, architecture). The most faithful way to create a
bundle is to dump the exact calibration/test arrays each `run_*.py` already computes,
so the bundle matches the published split, logits, and synthetic-annotation seed.

## Quick check (no cache needed)

```bash
/path/to/python analyze_depth.py --demo      # synthetic data; verifies the four diagnostics
```

## Real data: just pass `--dump-bundle`

The dump hook is now built into all four run scripts (flag-guarded; default off, so
normal runs are unchanged). Pull the logit cache first (`cache/` from the HF
checkpoints repo), then add `--dump-bundle` to any run:

```bash
python run_cifar10h.py   --arch resnet50         --dump-bundle
python run_cifar10h.py   --arch vit_b16          --dump-bundle
python run_chaosnli.py   --arch roberta_large    --dump-bundle
python run_chaosnli.py   --arch deberta_v3       --dump-bundle
python run_isic2019.py   --arch efficientnet_b4  --dump-bundle
python run_isic2019.py   --arch vit_s16          --dump-bundle
python run_dermamnist.py --arch resnet18         --dump-bundle
python run_dermamnist.py --arch vit_s16          --dump-bundle
```

Each writes `results/bundles/<dataset>_<arch>.npz` using the exact calibration/test
arrays that experiment computed (so the bundle matches the published split, logits,
and synthetic-annotation seed). For CIFAR-10H/ChaosNLI the calibration split is `*_cal`;
for ISIC/DermaMNIST it is the validation split and evaluation is the test split.

## Then run the analysis

```bash
python analyze_depth.py --bundle-dir results/bundles    # writes results/depth/*.json
```
This produces the D1–D4 diagnostics and the `v̄−ū`, `ρ`, `ΔECE` numbers for the
paper's Table `tab:lsts-valid` (Appendix N).

## Manual alternative (if you prefer not to use the flag)

`save_bundle(path, name, n_classes, logits_cal, logits_te, soft_cal, soft_te, hard_cal, hard_te)`
in `analyze_depth.py` can be called directly with any arrays (numpy or torch tensors).

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
