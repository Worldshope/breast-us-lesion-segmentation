# Breast Ultrasound Lesion Segmentation (nnU-Net v2 and UNet++)

Code and report for segmenting breast lesions in ultrasound images, developed to
approximate lesion localization on simulated phantom images. Two models were trained
on BUS-BRA and tested externally on BrEaST.

## Results

| Model | BUS-BRA internal, 5-fold CV (Dice) | BrEaST external (Dice) |
|---|---|---|
| nnU-Net v2, 2D, 320x320 | 91.14% | 76.76% (5-fold ensemble, no post-processing) |
| UNet++ (ResNet34), 256x256 | 87.40% ± 0.85% (std across folds) | 70.05% ± 29.28% (std across images; single model: fold-3 checkpoint, TTA + post-processing) |

nnU-Net external test (252 BrEaST images with lesions), other metrics:
IoU 67.33%, Precision 81.67%, Recall 78.22%, Specificity 98.11%, Accuracy 96.38%.
UNet++ external per-image distribution: median 83.45%, 25th/75th percentile 61.4%/90.4%,
Dice >= 0.8 in 59.5% of images, Dice < 0.5 in 21.4%.

**Inference time.** UNet++ (Google Colab, Tesla T4, 256x256, batch size 1, excluding image
loading): 24.8 ± 3.1 ms/image for one forward pass, 105.5 ± 4.9 ms/image with TTA (4 passes) and
post-processing. nnU-Net: can not reach results for now.

The two models differ in preprocessing (320x320 vs 256x256), loss, augmentation,
cross-validation splitting (random vs patient-wise) and inference tricks, so the comparison is
indicative and does not isolate the effect of the architecture.

## Repository structure
- `nnunet/`: nnU-Net v2 data preparation, training, evaluation and commands (`nnunet/commands.md`)
- `unetpp/`: UNet++ training, evaluation, inference time, plots (`common.py` must stay in this folder)
- `docs/`: technical report

## Datasets (not included)
- BUS-BRA: https://zenodo.org/records/8231412 (1875 images with masks)
- BrEaST-Lesions_USG: https://www.cancerimagingarchive.net/collection/breast-lesions-usg/ (256 images; 4 normal images have no mask)

## nnU-Net v2
Follow `nnunet/commands.md` (inference is the `nnUNetv2_predict` command in step 6).
Install: `pip install -r nnunet/requirements.txt` (install torch for your CUDA version first, see commands).
Edit the paths at the top of the scripts before running them.
Key settings: 2D configuration, default plans, 320x320 input, 5-fold CV, 1000 epochs,
5-fold ensemble at inference, no post-processing. Details: see the report.

## UNet++ (ResNet34)
Install: `pip install -r unetpp/requirements.txt`. `--busbra_dir` and `--breast_dir` must point to
the folder that directly contains `Images/` and `Masks/`.
```
python unetpp/train.py --busbra_dir /path/to/BUS-BRA --out_dir /path/to/output
python unetpp/evaluate.py --breast_dir /path/to/BrEaST --checkpoint /path/to/output/best_model_fold3.pth --out_dir /path/to/results
python unetpp/inference_time.py --breast_dir /path/to/BrEaST --checkpoint /path/to/output/best_model_fold3.pth --out_dir /path/to/results
python unetpp/plot_results.py --dices /path/to/results/breast_dices.npy --out_dir /path/to/results
```
`train.py` prints which fold is best; use that checkpoint for the other scripts.

## Using the trained models on new images
- **nnU-Net:** convert to grayscale, min-max normalize to 0-255, resize to 320x320 (bilinear),
  save as NIfTI with shape (H, W, 1). Resize the mask back with nearest-neighbor.
- **UNet++:** grayscale, resize to 256x256 (bilinear), divide by 255 (no min-max), threshold the
  sigmoid output at 0.5.
- **Scale matters.** With native image sizes the nnU-Net external Dice was 66.4%; after resizing
  both datasets to 320x320 it was 76.8%. If your images differ strongly in scale or appearance
  (e.g. phantom images), performance may drop. Check outputs visually.
- Trained weights are not included. Retraining nnU-Net takes several days.

## Known limitations
- nnU-Net used its default random 5-fold split, not verified to be patient-wise (left/right
  images of a patient may fall in different folds), so its internal CV may be slightly optimistic.
  The external result is unaffected.
- UNet++ folds are patient-wise, but the patient order comes from a Python `set`, so fold
  composition changes between runs, and no global random seed is set. A retraining gives slightly
  different results and the reported fold-3 checkpoint cannot be regenerated exactly.
- UNet++ internal Dice is the best epoch on the validation fold itself (also used for early
  stopping), so it is slightly optimistic. The external number comes from one checkpoint.
- UNet++ loss applies a sigmoid twice (model output and loss); kept as-is to match the reported results.
- All metrics are computed at the training resolution (nnU-Net 320x320, UNet++ 256x256), not the original one.
- nnU-Net precision is set to 1.0 when nothing is predicted, so mean precision can be optimistic
  for missed lesions. For nnU-Net only mean metrics were recorded, no per-image distribution.

## Contact
[FATMANUR BIRINCI, fatmanurbirincibirinci@gmail.com]