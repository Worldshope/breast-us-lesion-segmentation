"""UNet++ (ResNet34) - BrEaST external test (TTA + post-processing), per-image Dice saved.

Usage:
    python unetpp/evaluate.py --breast_dir <BrEaST root with Images/ and Masks/> \
        --checkpoint <best_model_foldN.pth> --out_dir <output folder>

Writes breast_dices.npy and breast_per_image_dice.csv to --out_dir.
Reported result: fold 3 checkpoint -> Dice 0.7005 ± 0.2928 (n = 252).
"""
import os
import csv
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (BUSDataset, build_model, list_breast_cases,
                    dice_score, tta_predict, post_process_mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--breast_dir", required=True, help="Folder that contains Images/ and Masks/ of BrEaST")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_area", type=int, default=100, help="Min component area in pixels (256x256 scale)")
    args = p.parse_args()

    images_dir = os.path.join(args.breast_dir, "Images")
    masks_dir = os.path.join(args.breast_dir, "Masks")
    os.makedirs(args.out_dir, exist_ok=True)

    breast_test_images, breast_test_mask_groups = list_breast_cases(images_dir, masks_dir)
    print(f"BrEaST images for test: {len(breast_test_images)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(encoder_weights=None).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    breast_dataset = BUSDataset(images_dir, masks_dir, breast_test_images, breast_test_mask_groups,
                                augment=False, is_breast=True, mask_groups=breast_test_mask_groups)
    breast_loader = DataLoader(breast_dataset, batch_size=1, shuffle=False)

    breast_dices = []
    print("\nTesting on BrEaST with TTA + Post-processing...")
    with torch.no_grad():
        for image, mask in tqdm(breast_loader, desc="BrEaST Test"):
            image = image.to(device)
            mask = mask.to(device)

            output = tta_predict(model, image)

            output_np = output[0, 0].cpu().numpy()
            processed = post_process_mask(output_np, min_area=args.min_area)
            output_processed = torch.from_numpy(processed).unsqueeze(0).to(device)

            breast_dices.append(dice_score(output_processed[0], mask[0]).item())

    dices = np.array(breast_dices)
    np.save(os.path.join(args.out_dir, "breast_dices.npy"), dices)
    with open(os.path.join(args.out_dir, "breast_per_image_dice.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "n_masks", "dice"])
        for name, group, d in zip(breast_test_images, breast_test_mask_groups, dices):
            w.writerow([name, len(group), f"{d:.6f}"])

    print(f"\nBrEaST Test Results (with TTA + Post-processing), n={len(dices)}:")
    print(f"Mean Dice: {dices.mean():.4f} ± {dices.std():.4f} ({dices.mean()*100:.2f}% ± {dices.std()*100:.2f}%)")
    print(f"Saved: breast_dices.npy, breast_per_image_dice.csv -> {args.out_dir}")


if __name__ == "__main__":
    main()
