# Breast Ultrasound Lesion Segmentation

This repository contains the code for breast ultrasound (BUS) lesion segmentation using two approaches:
1. **UNet++ with ResNet34 encoder** (custom implementation)
2. **nnU-Net v2** (auto-configured)

Both models were trained on the **BUS-BRA** dataset and externally validated on the **BrEaST** dataset.

## 📊 Results

| Model | BUS-BRA (Internal Validation) | BrEaST (External Test) |
|-------|-------------------------------|------------------------|
| UNet++ (ResNet34) | 87.40% ± 0.85% | 70.05% ± 29.28% |
| nnU-Net v2 | *To be updated* | **76.90%** |

*(Note: nnU-Net specific evaluation metrics and scripts will be added in the next commit.)*

## 📁 Repository Structure

- `unetpp/` — UNet++ training, inference, and evaluation scripts (ResNet34 encoder)
- `nnunet/` — nnU-Net preprocessing, training, and evaluation scripts
- `evaluation/` — Common evaluation metrics and visualization tools
- `docs/` — Technical report and presentation figures

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt

2. UNet++ Training & Evaluation
Before running the scripts, ensure your dataset directories are structured correctly (e.g., containing Images/ and Masks/ subfolders). Replace the placeholder paths (/path/to/...) with the actual paths on your local machine.

Training (5-fold cross-validation):

python unetpp/train.py --busbra_dir /path/to/BUS-BRA --out_dir /path/to/output_folder


External Testing (with TTA + Post-processing):

python unetpp/evaluate.py --breast_dir /path/to/BrEaST --checkpoint /path/to/output_folder/best_model_fold3.pth --out_dir /path/to/results

(Optional) To calculate and plot the summary statistics from the saved results:

python unetpp/plot_results.py --dices /path/to/results/breast_dices.npy --out_dir /path/to/results


3. nnU-Net Training & Evaluation
(nnU-Net scripts and instructions will be added soon)

Datasets
BUS-BRA: Kaggle

BrEaST: TCIA

Contact
Fatmanur Birinci/fatmanurbirincibirinci@gmail.com
