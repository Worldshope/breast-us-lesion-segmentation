#training_new.py

import subprocess
import os
import sys

# Environment variables
os.environ["nnUNet_raw"] = r"/data/users/dataset/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"/data/users/dataset/nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"/data/users/dataset/nnUNet_results"

# Configuration
DATASET_ID = "3"
CONFIGURATION = "2d"
FOLDS = [0, 1, 2, 3, 4]

print("="*60)
print("Training nnU-Net for 1000 epochs (Default) per fold")
print("="*60)

for fold in FOLDS:
    print(f"\n{'='*60}")
    print(f"Starting Fold {fold}/4")
    print(f"{'='*60}")

    cmd = f"nnUNetv2_train {DATASET_ID} {CONFIGURATION} {fold}"

    print(f"Command: {cmd}")
    print("-"*60)

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f" Error in fold {fold}! Stopping...")
        sys.exit(1)

    print(f" Fold {fold} completed successfully!")

print("\n" + "="*60)
print(" ALL FOLDS TRAINED SUCCESSFULLY!")
print("="*60)

print("\nFinding best configuration...")
subprocess.run(f"nnUNetv2_find_best_configuration {DATASET_ID} -c {CONFIGURATION}", shell=True)