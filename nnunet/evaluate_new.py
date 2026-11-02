#evaluate_new.py


import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    dice_denom = (2 * TP) + FP + FN
    dice = (2.0 * TP) / dice_denom if dice_denom > 0 else (1.0 if np.sum(y_true) == 0 and np.sum(y_pred) == 0 else 0.0)

    iou_denom = TP + FP + FN
    iou = TP / iou_denom if iou_denom > 0 else (1.0 if np.sum(y_true) == 0 and np.sum(y_pred) == 0 else 0.0)

    precision_denom = TP + FP
    precision = TP / precision_denom if precision_denom > 0 else 1.0

    recall_denom = TP + FN
    recall = TP / recall_denom if recall_denom > 0 else 1.0

    specificity_denom = TN + FP
    specificity = TN / specificity_denom if specificity_denom > 0 else 1.0

    accuracy_denom = TP + TN + FP + FN
    accuracy = (TP + TN) / accuracy_denom if accuracy_denom > 0 else 1.0

    return dice, iou, precision, recall, specificity, accuracy

gt_dir = r"/data/users/dataset/nnUNet_raw/Dataset003_BUSBRA_resized/labelsTs"
pred_dir = r"/data/users/dataset/nnUNet_predictions_v3"

pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]

metrics_list = {
    'dice': [], 'iou': [], 'precision': [],
    'recall': [], 'specificity': [], 'accuracy': []
}

print("=" * 60)
print("Calculating Comprehensive Metrics for nnU-Net External Test...")
print("=" * 60)

for f in tqdm(pred_files, desc="Evaluating"):
    pred_path = os.path.join(pred_dir, f)
    pred_img = nib.load(pred_path).get_fdata().astype(np.uint8)

    gt_path = os.path.join(gt_dir, f)
    if not os.path.exists(gt_path):
        print(f"Attention: for {f}, no real mask found, pass.")
        continue
    gt_img = nib.load(gt_path).get_fdata().astype(np.uint8)

    if pred_img.shape != gt_img.shape:
        print(f"Warning: {f} shape incompatibility- pred: {pred_img.shape}, gt: {gt_img.shape}")
        continue

    pred_binary = (pred_img > 0).astype(np.uint8)
    gt_binary = (gt_img > 0).astype(np.uint8)

    d, i, p, r, s, a = calculate_metrics(gt_binary, pred_binary)

    metrics_list['dice'].append(d)
    metrics_list['iou'].append(i)
    metrics_list['precision'].append(p)
    metrics_list['recall'].append(r)
    metrics_list['specificity'].append(s)
    metrics_list['accuracy'].append(a)

print("\n" + "=" * 60)
print(f"nnU-Net BrEaST External Test Final Results (Total: {len(metrics_list['dice'])} images):")
print("-" * 60)
print(f"Dice Score (F1) : %{np.mean(metrics_list['dice']) * 100:.2f}")
print(f"IoU (Jaccard)   : %{np.mean(metrics_list['iou']) * 100:.2f}")
print(f"Precision       : %{np.mean(metrics_list['precision']) * 100:.2f}")
print(f"Recall (Sens.)  : %{np.mean(metrics_list['recall']) * 100:.2f}")
print(f"Specificity     : %{np.mean(metrics_list['specificity']) * 100:.2f}")
print(f"Accuracy        : %{np.mean(metrics_list['accuracy']) * 100:.2f}")
print("=" * 60)

# Saving the results
import json
summary = {k: float(np.mean(v)) * 100 for k, v in metrics_list.items()}
summary['total_images'] = len(metrics_list['dice'])
with open("/data/users/dataset/evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=4)
print("\nResults saved: /data/users/dataset/evaluation_summary.json")