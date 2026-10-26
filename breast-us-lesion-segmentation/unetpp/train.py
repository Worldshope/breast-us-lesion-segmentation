"""UNet++ (ResNet34) - BUS-BRA 5-fold patient-wise cross-validation (Test 7 setting).

Saves best_model_fold{1..5}.pth, qualitative samples and cv_results.json in --out_dir.
The external test on BrEaST is done by evaluate.py.

Usage:
    python unetpp/train.py --busbra_dir <BUSBRA root with Images/ and Masks/> --out_dir <output folder>
"""
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold

from common import (BUSDataset, DiceFocalLoss, build_model, get_patient_id,
                    dice_score, calculate_per_class_dice)

# Training hyper-parameters (as used for the reported results)
K_FOLDS = 5
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--busbra_dir", required=True, help="Folder that contains Images/ and Masks/ of BUS-BRA")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    busbra_images_dir = os.path.join(args.busbra_dir, "Images")
    busbra_masks_dir = os.path.join(args.busbra_dir, "Masks")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- BUS-BRA file lists and patient ids
    image_files = sorted([f for f in os.listdir(busbra_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(busbra_masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_dict = {f: True for f in mask_files}

    patient_ids = [get_patient_id(f) for f in image_files]
    # NOTE: set() order is not fixed between Python runs, so the fold composition varies between runs
    # (see README, "Known limitations"). Kept as in the original experiments.
    unique_patients = list(set(patient_ids))
    print(f"Total BUS-BRA patients: {len(unique_patients)}")
    print(f"Total BUS-BRA images: {len(image_files)}")

    # ---- K-fold cross-validation (patient-wise)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_patients)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{K_FOLDS}")
        print(f"{'='*60}")

        train_patients = [unique_patients[i] for i in train_idx]
        val_patients = [unique_patients[i] for i in val_idx]

        train_images = [f for f in image_files if get_patient_id(f) in train_patients]
        val_images = [f for f in image_files if get_patient_id(f) in val_patients]

        train_masks = [f.replace("bus_", "mask_") for f in train_images if f.replace("bus_", "mask_") in mask_dict]
        val_masks = [f.replace("bus_", "mask_") for f in val_images if f.replace("bus_", "mask_") in mask_dict]

        print(f"Train: {len(train_images)} images, {len(train_patients)} patients")
        print(f"Val: {len(val_images)} images, {len(val_patients)} patients")

        train_dataset = BUSDataset(busbra_images_dir, busbra_masks_dir, train_images, train_masks, augment=True)
        val_dataset = BUSDataset(busbra_images_dir, busbra_masks_dir, val_images, val_masks, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = build_model(encoder_weights="imagenet").to(device)

        loss_fn = DiceFocalLoss(alpha=0.5, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_dice = 0
        best_fg_dice = 0
        best_bg_dice = 0
        patience = EARLY_STOP_PATIENCE
        counter = 0
        best_epoch = 0
        checkpoint_path = os.path.join(args.out_dir, f'best_model_fold{fold+1}.pth')

        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} - Training"):
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0
            val_dice = 0
            val_fg_dice = 0
            val_bg_dice = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
                    val_loss += loss.item()
                    for i in range(images.size(0)):
                        val_dice += dice_score(outputs[i], masks[i]).item()
                        fg_dice, bg_dice = calculate_per_class_dice(outputs[i], masks[i])
                        val_fg_dice += fg_dice
                        val_bg_dice += bg_dice

            avg_val_loss = val_loss / len(val_loader)
            avg_val_dice = val_dice / len(val_dataset)
            avg_fg_dice = val_fg_dice / len(val_dataset)
            avg_bg_dice = val_bg_dice / len(val_dataset)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Dice={avg_val_dice:.4f}, FG={avg_fg_dice:.4f}, BG={avg_bg_dice:.4f}, LR={current_lr:.2e}")

            scheduler.step(avg_val_loss)

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                best_fg_dice = avg_fg_dice
                best_bg_dice = avg_bg_dice
                counter = 0
                best_epoch = epoch + 1
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  >>> Best model saved with Dice: {best_val_dice:.4f}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered. Best Dice: {best_val_dice:.4f} at epoch {best_epoch}")
                    break

        # Qualitative samples
        samples_dir = os.path.join(args.out_dir, f'fold_{fold+1}_samples')
        os.makedirs(samples_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            images, masks = next(iter(val_loader))
            images = images.to(device)
            outputs = model(images)
            for i in range(min(4, len(images))):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(images[i, 0].cpu(), cmap='gray')
                axes[0].set_title("Original")
                axes[0].axis('off')
                axes[1].imshow(masks[i, 0].cpu(), cmap='gray')
                axes[1].set_title("Ground Truth")
                axes[1].axis('off')
                axes[2].imshow(outputs[i, 0].cpu() > 0.5, cmap='gray')
                axes[2].set_title("Prediction")
                axes[2].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(samples_dir, f'sample_{i}.png'))
                plt.close()

        fold_results.append({
            'fold': fold + 1,
            'best_dice': best_val_dice,
            'best_fg_dice': best_fg_dice,
            'best_bg_dice': best_bg_dice,
            'best_epoch': best_epoch,
        })

    # ---- K-fold results
    print("\n" + "=" * 60)
    print("5-FOLD CV RESULTS")
    print("=" * 60)

    dice_scores = []
    for res in fold_results:
        print(f"Fold {res['fold']}: Dice={res['best_dice']:.4f}, FG={res['best_fg_dice']:.4f}, BG={res['best_bg_dice']:.4f}")
        dice_scores.append(res['best_dice'])

    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    print(f"\nMean Dice: {mean_dice:.4f} ± {std_dice:.4f} ({mean_dice*100:.2f}% ± {std_dice*100:.2f}%)")

    best_fold_idx = int(np.argmax(dice_scores))
    best_fold = fold_results[best_fold_idx]['fold']
    print(f"\nBest fold: {best_fold} (Dice: {dice_scores[best_fold_idx]:.4f})")
    print(f"Run the external test with: --checkpoint {os.path.join(args.out_dir, f'best_model_fold{best_fold}.pth')}")

    with open(os.path.join(args.out_dir, "cv_results.json"), "w") as f:
        json.dump({
            "folds": fold_results,
            "mean_dice": float(mean_dice),
            "std_dice": float(std_dice),
            "best_fold": best_fold,
            "best_checkpoint": f"best_model_fold{best_fold}.pth",
        }, f, indent=2)


if __name__ == "__main__":
    main()
