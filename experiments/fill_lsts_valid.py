"""
One-shot filler for the paper's Table tab:lsts-valid (LS-TS proxy diagnostics).

The 8 published rows need the ORIGINAL cached logits. Once the logit cache is
restored, the full pipeline is three commands:

    # 1) dump a bundle for each published (dataset, arch) -- the --dump-bundle hook
    #    writes results/bundles/<dataset>_<arch>.npz using the exact published split:
    python run_cifar10h.py   --arch resnet50        --dump-bundle
    python run_cifar10h.py   --arch vit_b16         --dump-bundle
    python run_chaosnli.py   --arch roberta_large   --dump-bundle
    python run_chaosnli.py   --arch deberta_v3      --dump-bundle
    python run_isic2019.py   --arch efficientnet_b4 --dump-bundle
    python run_isic2019.py   --arch vit_s16         --dump-bundle
    python run_dermamnist.py --arch resnet18        --dump-bundle
    python run_dermamnist.py --arch vit_s16         --dump-bundle

    # 2) compute the depth diagnostics (D1-D4) for every bundle:
    python analyze_depth.py --bundle-dir results/bundles --out results/depth

    # 3) emit the paste-ready LaTeX rows for tab:lsts-valid:
    python fill_lsts_valid.py --depth-dir results/depth

This script reads the depth_*.json files and prints, per setting, the LaTeX row
    Dataset & Arch & g & rho & T_SLTS & T_LS & DeltaECE \\
where g = vbar(1-1/K)-ubar (K-corrected proxy gap, D4), rho = corr(u,v),
T_SLTS / T_LS are the fitted soft / LS-TS temperatures, and DeltaECE is the
LS-TS-minus-SLTS gap in ECE_true (population, D2) in percentage points.

Verify it works without the cache:  python fill_lsts_valid.py --depth-dir results/real
"""

import argparse, glob, json, os

# bundle/file stem -> (Dataset, Arch) label for the tab:lsts-valid rows.
NAME_MAP = {
    "cifar10h_resnet50":        ("CIFAR-10H", "R50"),
    "cifar10h_vit_b16":         ("CIFAR-10H", "ViT-B/16"),
    "chaosnli_roberta_large":   ("ChaosNLI", "RoBERTa"),
    "chaosnli_deberta_v3":      ("ChaosNLI", "DeBERTa"),
    "isic2019_efficientnet_b4": ("ISIC 2019", "ENet-B4"),
    "isic2019_vit_s16":         ("ISIC 2019", "ViT-S/16"),
    "dermamnist_resnet18":      ("DermaMNIST", "R18"),
    "dermamnist_vit_s16":       ("DermaMNIST", "ViT-S/16"),
    # real-data settings (for verification / App. O):
    "cifar10h_vit_real":        ("CIFAR-10H*", "ViT"),
    "cifar10h_cnn_real":        ("CIFAR-10H*", "R18"),
    "imagenet_real_resnet50":   ("ImageNet-ReaL", "R50"),
    "lidc_real":                ("LIDC-IDRI", "R18"),
    "nli_llm_real":             ("ChaosNLI-LLM", "Qwen-1.5B"),
    "ambignq_llm_real":         ("AmbigNQ-LLM", "Qwen-1.5B"),
}
# canonical paper order for the 8 published rows
ORDER = ["cifar10h_resnet50", "cifar10h_vit_b16", "chaosnli_roberta_large",
         "chaosnli_deberta_v3", "isic2019_efficientnet_b4", "isic2019_vit_s16",
         "dermamnist_resnet18", "dermamnist_vit_s16"]


def stem_of(path):
    b = os.path.basename(path)
    return b[len("depth_"):-len(".json")] if b.startswith("depth_") and b.endswith(".json") else b


def row(stem, r):
    d4 = r["D4_lsts_proxy"]; d2 = r["D2_confidence_bins"]
    g = d4["proxy_gap_K_corrected"]; rho = d4["rho_u_v"]
    ts, tl = d4["T_slts"], d4["T_lsts"]
    dece = None
    if "LS-TS" in d2 and "SLTS" in d2:
        dece = d2["LS-TS"]["ece_true_pop"] - d2["SLTS"]["ece_true_pop"]
    ds, arch = NAME_MAP.get(stem, (stem, ""))
    dece_s = f"{dece:+.2f}" if dece is not None else "--"
    return (f"    {ds:<14} & {arch:<10} & ${g:+.3f}$ & ${rho:.2f}$ & "
            f"${ts:.2f}$ & ${tl:.2f}$ & ${dece_s}$ \\\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-dir", default="results/depth",
                    help="dir of analyze_depth depth_*.json files")
    args = ap.parse_args()
    files = {stem_of(p): p for p in glob.glob(os.path.join(args.depth_dir, "depth_*.json"))}
    if not files:
        raise SystemExit(f"no depth_*.json in {args.depth_dir} — run analyze_depth.py first")
    keys = [k for k in ORDER if k in files] + [k for k in files if k not in ORDER]
    print("% paste into tab:lsts-valid  (cols: Dataset & Arch & g & rho & T_SLTS & T_LS & DeltaECE)")
    for k in keys:
        try:
            print(row(k, json.load(open(files[k]))))
        except Exception as e:
            print(f"    % {k}: ERROR {type(e).__name__}: {e}")
    missing = [k for k in ORDER if k not in files]
    if missing:
        print(f"\n% MISSING (run --dump-bundle + analyze_depth for these): {', '.join(missing)}")


if __name__ == "__main__":
    main()
