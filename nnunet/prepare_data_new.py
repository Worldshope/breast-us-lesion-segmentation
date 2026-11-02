# prepare_data_new.py

import os
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
import glob

# ============================================
# PATHS - UPDATE THESE
# ============================================
busbra_images = r"/data/users/dataset/Datasets/BUS-BRA_Main Dataset/BUSBRA/BUSBRA/Images"
busbra_masks = r"/data/users/dataset/Datasets/BUS-BRA_Main Dataset/BUSBRA/BUSBRA/Masks"

breast_images = r"/data/users/dataset/Datasets/BrEaST-Lesions_USG-images/BrEaST-Lesions_USG-images_and_masks/Images"
breast_masks = r"/data/users/dataset/Datasets/BrEaST-Lesions_USG-images/BrEaST-Lesions_USG-images_and_masks/Masks"

# nnU-Net directories
nnunet_raw = r"/data/users/dataset/nnUNet_raw"


DATASET_ID = 3
DATASET_NAME = "BUSBRA_resized"
dataset_path = os.path.join(nnunet_raw, f"Dataset{DATASET_ID:03d}_{DATASET_NAME}")

# Create folder structure
for subfolder in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
    os.makedirs(os.path.join(dataset_path, subfolder), exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
TARGET_H, TARGET_W = 320, 320 #resized

def save_as_nifti(img_array, output_path, is_mask=False):
    if is_mask:
        data = img_array.astype(np.uint8)
        interp = cv2.INTER_NEAREST   # nearest-neighbor for the mask
    else:
        data = img_array.astype(np.float32)
        interp = cv2.INTER_LINEAR    # bilinear for the image

    # --- STANDARDIZE RESIZE ---
    data = cv2.resize(data, (TARGET_W, TARGET_H), interpolation=interp)

    # Post-resize binarization for safety to ensure absolutely 0 and 1
    if is_mask:
        data = (data > 0).astype(np.uint8)

    if len(data.shape) == 2:
        data = data[..., None]  # (H, W) -> (H, W, 1)

    img_nifti = nib.Nifti1Image(data.astype(np.uint8 if is_mask else np.float32), np.eye(4))
    nib.save(img_nifti, output_path)

def normalize_image(img):
    """Normalize image to [0, 255] range."""
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img

# ============================================
# 1. CONVERT BUS-BRA (TRAINING SET)
# ============================================
print("=" * 60)
print("STEP 1: Converting BUS-BRA dataset to NIfTI")
print("=" * 60)

busbra_img_files = sorted([f for f in os.listdir(busbra_images) if f.endswith('.png')])
busbra_mask_files = set(os.listdir(busbra_masks))

converted_count = 0
missing_masks = []

for img_f in tqdm(busbra_img_files, desc="BUS-BRA conversion"):
    mask_f = img_f.replace("bus_", "mask_")

    if mask_f in busbra_mask_files:
        case_id = img_f.split(".")[0].replace("-", "_")

        # Read and normalize image
        img = cv2.imread(os.path.join(busbra_images, img_f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_f}")
            continue
        img = normalize_image(img)
        save_as_nifti(img, os.path.join(dataset_path, f"imagesTr/{case_id}_0000.nii.gz"))

        # Read mask (keep 0-255 during load, let function handle interpolation)
        mask = cv2.imread(os.path.join(busbra_masks, mask_f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read {mask_f}")
            continue

        save_as_nifti(mask, os.path.join(dataset_path, f"labelsTr/{case_id}.nii.gz"), is_mask=True)
        converted_count += 1
    else:
        missing_masks.append(img_f)

print(f"Converted {converted_count}/{len(busbra_img_files)} BUS-BRA images")

# ============================================
# 2. CONVERT BrEaST (EXTERNAL TEST SET)
# ============================================
print("\n" + "=" * 60)
print("STEP 2: Converting BrEaST dataset to NIfTI")
print("=" * 60)

breast_img_files = sorted([f for f in os.listdir(breast_images) if f.endswith('.png')])
breast_mask_files = sorted([f for f in os.listdir(breast_masks) if f.endswith('.png')])

mask_dict = {}
for mask_f in breast_mask_files:
    case_num = mask_f.split("_")[0] if "_" in mask_f else mask_f.split(".")[0]
    if case_num not in mask_dict:
        mask_dict[case_num] = []
    mask_dict[case_num].append(mask_f)

converted_count = 0
multi_mask_cases = 0

for img_f in tqdm(breast_img_files, desc="BrEaST conversion"):
    base_name = img_f.split(".")[0]
    output_case_id = f"breast_{base_name}"

    # Read and normalize image
    img = cv2.imread(os.path.join(breast_images, img_f), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read {img_f}")
        continue
    img = normalize_image(img)
    save_as_nifti(img, os.path.join(dataset_path, f"imagesTs/{output_case_id}_0000.nii.gz"))

    # Combine all masks for this image (keep 0-255 range for clean scaling)
    combined_mask = np.zeros_like(img, dtype=np.uint8)

    if base_name in mask_dict:
        for mask_f in mask_dict[base_name]:
            mask = cv2.imread(os.path.join(breast_masks, mask_f), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                combined_mask = np.maximum(combined_mask, mask)

        if combined_mask.sum() > 0:
            save_as_nifti(combined_mask, os.path.join(dataset_path, f"labelsTs/{output_case_id}.nii.gz"), is_mask=True)
            converted_count += 1
            if len(mask_dict[base_name]) > 1:
                multi_mask_cases += 1

print(f"Converted {converted_count}/{len(breast_img_files)} BrEaST images")

# ============================================
# 3. CREATE dataset.json
# ============================================
num_training = len([f for f in os.listdir(os.path.join(dataset_path, "labelsTr")) if f.endswith('.nii.gz')])
num_test = len([f for f in os.listdir(os.path.join(dataset_path, "labelsTs")) if f.endswith('.nii.gz')])

dataset_json = {
    "name": DATASET_NAME,
    "description": "BUS-BRA breast ultrasound lesion segmentation dataset standardized to 320x320",
    "reference": "Internal Work",
    "licence": "Academic use only",
    "release": "1.0",
    "numTraining": num_training,
    "numTest": num_test,
    "channel_names": {
        "0": "ultrasound"
    },
    "labels": {
        "background": 0,
        "lesion": 1
    },
    "modality": {
        "0": "US"
    },
    "file_ending": ".nii.gz"
}

with open(os.path.join(dataset_path, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"\n[SUCCESS] Data preparation completed for Dataset003: {dataset_path}")

mask = nib.load("/data/users/dataset/nnUNet_raw/Dataset003_BUSBRA_resized/labelsTr/bus_0001_l.nii.gz").get_fdata()
print("Unique Values:", np.unique(mask))