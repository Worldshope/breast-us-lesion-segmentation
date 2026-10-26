"""Shared code for the UNet++ (ResNet34) baseline: data, augmentation, metrics, TTA, post-processing, loss.

All logic is identical to the Test 7 training script that produced the reported results.
Only file paths and the split into several files were changed.
"""
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------- augmentation helpers
def add_speckle_noise(image, std=0.05):
    noise = np.random.normal(0, std, image.shape)
    speckle = image + noise * image
    return np.clip(speckle, 0, 1)


def gamma_correction(image, gamma=1.2):
    return np.power(image, gamma)


def apply_clahe(image, clip_limit=2.0, grid_size=8):
    img = (image[0] * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(img)
    enhanced = enhanced.astype(np.float32) / 255.0
    return np.expand_dims(enhanced, axis=0)


# ---------------------------------------------------------------- BUS-BRA patient id
def get_patient_id(filename):
    """bus_0001-l.png -> '0001' (left and right breast of a patient share the id)."""
    parts = filename.split('_')
    if len(parts) >= 2:
        patient_num = parts[1].split('-')[0]
        return patient_num
    return None


# ---------------------------------------------------------------- BrEaST file pairing
def list_breast_cases(images_dir, masks_dir):
    """Images that have at least one mask, and the list of masks of each image.

    case036.png has masks case036_tumor.png, case036_other1.png, ...
    Images without any mask (the 4 normal cases) are skipped automatically.
    """
    breast_image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    breast_mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

    image_to_masks = defaultdict(list)
    for mask in breast_mask_files:
        base_name = mask.split('_')[0] + '.png'
        if base_name in breast_image_files:
            image_to_masks[base_name].append(mask)

    breast_test_images = list(image_to_masks.keys())
    breast_test_mask_groups = [image_to_masks[img] for img in breast_test_images]
    return breast_test_images, breast_test_mask_groups


# ---------------------------------------------------------------- dataset
class BUSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_files, mask_files, size=256, augment=False, is_breast=False, mask_groups=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.mask_files = mask_files
        self.size = size
        self.augment = augment
        self.is_breast = is_breast
        self.mask_groups = mask_groups

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        if self.is_breast and self.mask_groups:
            # all masks of the image (tumor + other lesions) are merged into one binary mask
            combined_mask = np.zeros((self.size, self.size), dtype=np.float32)
            for mask_name in self.mask_groups[idx]:
                mask_path = os.path.join(self.mask_dir, mask_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.size, self.size))
                mask = mask.astype(np.float32) / 255.0
                mask = (mask > 0.5).astype(np.float32)
                combined_mask = np.maximum(combined_mask, mask)
            mask = np.expand_dims(combined_mask, axis=0)
        else:
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.size, self.size))
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)

        # AUGMENTATION (training only)
        if self.augment:
            # Rotation (multiples of 90 degrees)
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(1, 2)).copy()
            # Horizontal flip
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()
            # Vertical flip
            if np.random.rand() > 0.3:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            # Speckle noise (ultrasound)
            if np.random.rand() > 0.3:
                noise_std = np.random.uniform(0.02, 0.1)
                image = add_speckle_noise(image, std=noise_std)
            # Gamma correction
            if np.random.rand() > 0.5:
                gamma = np.random.uniform(0.7, 1.5)
                image = gamma_correction(image, gamma=gamma)
            # CLAHE (contrast)
            if np.random.rand() > 0.6:
                clip_limit = np.random.uniform(1.5, 3.0)
                image = apply_clahe(image, clip_limit=clip_limit)
            # Brightness
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.7, 1.3)
                image = np.clip(image * brightness, 0, 1)
            # Gaussian blur
            if np.random.rand() > 0.7:
                blur_ksize = np.random.choice([3, 5])
                image_blurred = cv2.GaussianBlur(image[0], (blur_ksize, blur_ksize), 0)
                image = np.expand_dims(image_blurred, axis=0)

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()


# ---------------------------------------------------------------- metrics
def dice_score(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    return (2. * intersection) / (pred_binary.sum() + target.sum() + 1e-8)


def calculate_per_class_dice(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    tp_fg = (pred_binary * target_binary).sum()
    fg_dice = (2 * tp_fg) / (pred_binary.sum() + target_binary.sum() + 1e-8)
    pred_bg = 1 - pred_binary
    target_bg = 1 - target_binary
    tp_bg = (pred_bg * target_bg).sum()
    bg_dice = (2 * tp_bg) / (pred_bg.sum() + target_bg.sum() + 1e-8)
    return fg_dice.item(), bg_dice.item()


# ---------------------------------------------------------------- TTA and post-processing
def tta_predict(model, image):
    """Test-time augmentation: original + h-flip + v-flip + rot90, probabilities averaged."""
    preds = []
    preds.append(model(image))
    flipped_h = torch.flip(image, dims=[3])
    preds.append(torch.flip(model(flipped_h), dims=[3]))
    flipped_v = torch.flip(image, dims=[2])
    preds.append(torch.flip(model(flipped_v), dims=[2]))
    rotated = torch.rot90(image, k=1, dims=[2, 3])
    preds.append(torch.rot90(model(rotated), k=-1, dims=[2, 3]))
    return torch.stack(preds).mean(dim=0)


def post_process_mask(mask, min_area=100, threshold=0.5):
    """Remove connected components smaller than min_area (pixels at 256x256), then 3x3 closing."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(binary_mask, [cnt], -1, 0, -1)
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return np.expand_dims(binary_mask.astype(np.float32), axis=0)


# ---------------------------------------------------------------- loss and model
class DiceFocalLoss(nn.Module):
    """NOTE: the model already outputs sigmoid probabilities, while smp losses apply a sigmoid
    internally (double sigmoid). Kept as-is because all reported results were trained this way."""

    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.focal = smp.losses.FocalLoss(mode='binary', alpha=alpha, gamma=gamma)

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.focal(y_pred, y_true)


def build_model(encoder_weights):
    """UNet++ with ResNet34 encoder, 1 input channel, 1 output channel, sigmoid output."""
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,   # "imagenet" for training, None when loading a checkpoint
        in_channels=1,
        classes=1,
        activation="sigmoid",
    )
