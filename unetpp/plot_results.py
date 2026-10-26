"""Summary statistics and plots computed from saved results only.

Usage:
    python unetpp/plot_results.py --dices <breast_dices.npy> --out_dir <folder> [--cv_results cv_results.json]
If --cv_results (written by train.py) is not given, the fold Dice values reported in the report are used.
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTED_FOLD_DICE = [0.8800, 0.8575, 0.8804, 0.8777, 0.8744]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dices", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cv_results", default=None)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.cv_results:
        with open(args.cv_results) as f:
            fold_dice = [r["best_dice"] for r in json.load(f)["folds"]]
    else:
        fold_dice = REPORTED_FOLD_DICE

    dices = np.load(args.dices)
    cv_mean, cv_std = np.mean(fold_dice), np.std(fold_dice)

    lines = [
        "UNet++ (ResNet34) - summary", "=" * 60,
        "BUS-BRA 5-fold CV (best-epoch validation Dice)",
        f"  Folds: {[round(x, 4) for x in fold_dice]}",
        f"  Mean ± std: {cv_mean:.4f} ± {cv_std:.4f}  ({cv_mean*100:.2f}% ± {cv_std*100:.2f}%)",
        f"  Min / Max: {min(fold_dice):.4f} / {max(fold_dice):.4f}", "",
        f"BrEaST external test (TTA + post-processing), n = {len(dices)} images",
        f"  Mean ± std: {dices.mean():.4f} ± {dices.std():.4f}  ({dices.mean()*100:.2f}% ± {dices.std()*100:.2f}%)",
        f"  Median: {np.median(dices):.4f}",
        f"  Min / Max: {dices.min():.4f} / {dices.max():.4f}",
        f"  25th / 75th percentile: {np.percentile(dices, 25):.4f} / {np.percentile(dices, 75):.4f}",
        f"  Dice >= 0.8: {(dices >= 0.8).sum()} ({(dices >= 0.8).mean()*100:.1f}%)",
        f"  Dice < 0.5 : {(dices < 0.5).sum()} ({(dices < 0.5).mean()*100:.1f}%)",
        "  (std = population std, np.std with ddof=0; CV std is across folds, external std across images)",
    ]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "summary_stats.txt"), "w") as f:
        f.write(text + "\n")

    folds = np.arange(1, len(fold_dice) + 1)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(folds, fold_dice, color="steelblue", edgecolor="black")
    plt.axhline(cv_mean, color="red", linestyle="--", label=f"Mean = {cv_mean:.4f}")
    plt.fill_between([0.5, len(fold_dice) + 0.5], cv_mean - cv_std, cv_mean + cv_std, alpha=0.2,
                     color="red", label=f"±1 std = {cv_std:.4f}")
    for b, s in zip(bars, fold_dice):
        plt.text(b.get_x() + b.get_width() / 2, s + 0.002, f"{s:.4f}", ha="center", fontsize=9)
    plt.xlabel("Fold"); plt.ylabel("Best-epoch validation Dice")
    plt.title("BUS-BRA 5-fold cross-validation (UNet++)")
    plt.xticks(folds); plt.ylim(0.82, 0.90); plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "5fold_cv_results.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(dices, bins=30, range=(0, 1), color="coral", edgecolor="black", alpha=0.8)
    plt.axvline(dices.mean(), color="red", linestyle="--", label=f"Mean = {dices.mean():.4f}")
    plt.axvline(np.median(dices), color="blue", linestyle="--", label=f"Median = {np.median(dices):.4f}")
    plt.xlabel("Per-image Dice"); plt.ylabel("Number of images")
    plt.title(f"BrEaST external test, per-image Dice (n={len(dices)})")
    plt.legend(loc="upper left"); plt.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "breast_dice_distribution.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
