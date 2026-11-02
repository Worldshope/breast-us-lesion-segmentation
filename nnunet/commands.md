# nnU-Net v2 pipeline: commands used

Adjust paths and the CUDA version to your machine.

## 1. Environment
python3 -m venv nnunet_env
source nnunet_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2

## 2. Folders and environment variables (add to ~/.bashrc, then `source ~/.bashrc`)
mkdir -p /data/users/dataset/nnUNet_raw /data/users/dataset/nnUNet_preprocessed /data/users/dataset/nnUNet_results
export nnUNet_raw="/data/users/dataset/nnUNet_raw"
export nnUNet_preprocessed="/data/users/dataset/nnUNet_preprocessed"
export nnUNet_results="/data/users/dataset/nnUNet_results"

## 3. Prepare data (PNG -> NIfTI, 320x320; creates Dataset003_BUSBRA_resized)
python prepare_data_new.py

## 4. Plan and preprocess
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity

## 5. Training (5 folds, 1000 epochs each, 2d configuration)
python training_new.py

## 6. Inference on the external test set (5-fold ensemble)
mkdir -p /data/users/dataset/nnUNet_predictions_resized
nnUNetv2_predict \
  -i /data/users/dataset/nnUNet_raw/Dataset003_BUSBRA_resized/imagesTs \
  -o /data/users/dataset/nnUNet_predictions_resized \
  -d Dataset003_BUSBRA_resized \
  -c 2d -f 0 1 2 3 4 -tr nnUNetTrainer -p nnUNetPlans

## 7. Evaluation
python evaluate_new.py
