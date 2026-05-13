# UDAC Medical Image Segmentation Training Code

This repository contains PyTorch-based training scripts for semi-supervised medical image segmentation. The project is designed for UDAC-related experiments and includes several training variants based on UCMT, Mean Teacher, DCT-based learning, and multi-model collaborative training.

All scripts are assumed to be placed in the same project folder. Therefore, this README describes how to run the code directly from the current folder without requiring an additional recommended directory structure.

---

## 1. Project Overview

This project focuses on semi-supervised medical image segmentation. It supports both 2D and 3D segmentation experiments.

The main functions include:

- Binary medical image segmentation;
- Semi-supervised training with labeled and unlabeled data;
- 2D medical image segmentation, such as ISIC skin lesion segmentation and Kvasir-CVC polyp segmentation;
- 3D medical image segmentation, such as LA dataset segmentation;
- Segmentation backbones including DeepLabv3+, UNet, and VNet;
- UDAC / UCMT-style collaborative training;
- Mean Teacher semi-supervised learning;
- DCT-based frequency-domain learning;
- CutMix / BoxMask data augmentation;
- TensorBoard logging;
- Model checkpoint saving.

---

## 2. Code Files

The main training files are listed below.

| File | Description |
|---|---|
| `train.py` | Main 2D semi-supervised segmentation training script. It contains UCMT / DCT / multi-model related training logic. |
| `train1.py` | Experimental variant of the main 2D training script. |
| `train1_UADC.py` | UDAC-related training script. This file can be used as the main script for UDAC experiments. |
| `train_ucmt_isic.py` | UCMT training script for ISIC-style datasets. |
| `train_UCMT.py` | UCMT training script for polyp segmentation datasets such as Kvasir-CVC. |
| `train_kvasir_cvc.py` | Training script for Kvasir-CVC polyp segmentation. |
| `train1_kvasir_cvc.py` | Another experimental Kvasir-CVC training script. |
| `train_meanteacher.py` | Mean Teacher semi-supervised training script. |
| `train_3d.py` | 3D medical image segmentation training script using VNet by default. |
| `train_3d_6model.py` | 3D multi-model collaborative training script. |
| `train_piture.py` | Image experiment or visualization-related training variant. The file name may be intended as `train_picture.py`. |

---

## 3. Environment Requirements

A Linux system with an NVIDIA GPU is recommended.

### 3.1 Basic Environment

```bash
python >= 3.8
CUDA >= 11.x
PyTorch >= 1.10
```

### 3.2 Python Dependencies

Install the required Python packages:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm easydict pyyaml opencv-python tensorboard
pip install unfoldNd
```

Depending on the specific script, the following local modules should also be available in the same project folder or accessible by Python:

```text
data/
dataset/
models/
model/
utils/
util/
supervised.py
supervised_dct.py
```

---

## 4. Dataset Preparation

The scripts support different dataset formats depending on the experiment.

### 4.1 ISIC / 2D Medical Image Dataset

Applicable scripts include:

```text
train.py
train1.py
train1_UADC.py
train_ucmt_isic.py
train_meanteacher.py
train_piture.py
```

These scripts usually use:

```bash
--data_path YOUR_DATA_PATH
```

A common dataset format is:

```text
YOUR_DATA_PATH/
├── train/
│   ├── image/
│   └── mask/
├── val/
│   ├── image/
│   └── mask/
└── test/
    ├── image/
    └── mask/
```

The actual folder names should match the implementation of `ISICDataset` in `data/dataset.py`.

### 4.2 Kvasir-CVC Dataset

Applicable scripts include:

```text
train_kvasir_cvc.py
train1_kvasir_cvc.py
train_UCMT.py
```

These scripts usually use:

```bash
--image_root PATH_TO_IMAGES
--gt_root PATH_TO_MASKS
```

A common dataset format is:

```text
TrainDataset_cvc_kvasir/
├── image/
│   ├── xxx.png
│   └── ...
└── mask/
    ├── xxx.png
    └── ...
```

### 4.3 3D LA Dataset

Applicable scripts include:

```text
train_3d.py
train_3d_6model.py
```

These scripts usually use:

```bash
--data_path YOUR_3D_DATA_PATH
```

The default 3D input size is:

```text
[80, 112, 112]
```

The actual data loading process depends on the implementation of `LADataset` in `data/dataset_3d.py`.

---

## 5. Quick Start

All commands can be executed directly in the current project folder.

### 5.1 Run UDAC Training

```bash
python train1_UADC.py   --data_path ./dataset/ISIC/   --backbone DeepLabv3p   --image_size 256   --labeled_percentage 0.05   --num_epochs 25   --batch_size 4   --learning_rate 5e-4
```

### 5.2 Run Main 2D Training

```bash
python train.py   --data_path ./dataset/ISIC/   --labeled_percentage 0.05   --num_epochs 25   --batch_size 4   --learning_rate 5e-4   --config ./configs/pascal/segformerb2_4x4.yaml   --config2 ./configs/pascal/r50_dct_4x4.yaml   --config3 ./configs/pascal/r50_4x4.yaml
```

### 5.3 Run UCMT on ISIC

```bash
python train_ucmt_isic.py   --data_path ./dataset/ISIC/   --backbone DeepLabv3p   --image_size 256   --labeled_percentage 0.1   --num_epochs 25   --batch_size 4   --learning_rate 1e-4
```

### 5.4 Run Mean Teacher Training

```bash
python train_meanteacher.py   --data_path ./dataset/ISIC/   --backbone DeepLabv3p   --labeled_percentage 0.05   --num_epochs 25   --batch_size 4
```

### 5.5 Run Kvasir-CVC Training

```bash
python train_kvasir_cvc.py   --image_root ./dataset/TrainDataset_cvc_kvasir/image/   --gt_root ./dataset/TrainDataset_cvc_kvasir/mask/   --backbone DeepLabv3p   --image_size 256   --labeled_percentage 0.15   --num_epochs 25   --batch_size 4   --learning_rate 5e-4   --config ./configs/pascal/segformerb2_4x4.yaml   --config2 ./configs/pascal/r50_dct_4x4.yaml   --config3 ./configs/pascal/r50_4x4.yaml
```

### 5.6 Run 3D Training

```bash
python train_3d.py   --data_path ./dataset/LA/   --backbone VNet   --labeled_percentage 0.1   --num_epochs 1000   --batch_size 4   --learning_rate 1e-4
```

---

## 6. Common Parameters

| Parameter | Default Value | Description |
|---|---:|---|
| `--seed` | `1` | Random seed. |
| `--project` | `./runs/UCMT` | Directory for saving experiment results. |
| `--backbone` | `DeepLabv3p` / `UNet` / `VNet` | Segmentation backbone. |
| `--data_path` | `YOUR_DATA_PATH` | Dataset path for ISIC-style or 3D datasets. |
| `--image_root` | `YOUR_DATA_PATH` | Image path for Kvasir-CVC-style datasets. |
| `--gt_root` | `YOUR_DATA_PATH` | Ground-truth mask path for Kvasir-CVC-style datasets. |
| `--image_size` | `256` or `[80,112,112]` | Input image or volume size. |
| `--labeled_percentage` | `0.05`, `0.1`, `0.15`, `0.3` | Percentage of labeled data used in semi-supervised training. |
| `--is_cutmix` | `False` | Whether to enable CutMix / BoxMask. |
| `--mix_prob` | `0.5` | Probability for amplitude mixing or augmentation. |
| `--topk` | `1` or `2` | Top-k selection setting. |
| `--num_epochs` | `25`, `50`, `1000` | Number of training epochs. |
| `--batch_size` | `4` | Batch size. |
| `--num_workers` | `2` | Number of workers used by DataLoader. |
| `--in_channels` | `3` or `1` | Number of input channels. 2D RGB images use 3 channels; 3D volumes usually use 1 channel. |
| `--num_classes` | `2` | Number of segmentation classes. |
| `--pretrained` | `True` | Whether to use pretrained weights. |
| `--learning_rate` | `1e-4` or `5e-4` | Learning rate. |
| `--config` | YAML path | Configuration file for the first model. |
| `--config2` | YAML path | Configuration file for the DCT branch or second model. |
| `--config3` | YAML path | Configuration file for the third model. |
| `--thr` | `0.95` | Confidence threshold for pseudo-labels. |
| `--uw` | `1.0` | Weight of the unsupervised loss. |
| `--amp` | `False` | Whether to enable automatic mixed precision training. |

---

## 7. Training Pipeline

The general training process is:

1. Parse command-line arguments;
2. Set the random seed;
3. Create the experiment output folder;
4. Load the training and validation datasets;
5. Split the training data into labeled and unlabeled subsets according to `labeled_percentage`;
6. Repeat the labeled subset to balance the number of labeled and unlabeled samples;
7. Build DataLoaders;
8. Initialize segmentation models;
9. Define the segmentation loss, such as Dice / DSC loss;
10. Define the optimizer, such as AdamW or SGD;
11. Train the model using supervised loss, unsupervised consistency loss, pseudo-label loss, or DCT-related loss;
12. Record logs using TensorBoard and log files;
13. Save model checkpoints.

---

## 8. Output Results

Training results are saved under the folder specified by `--project`.

By default, the output directory is usually similar to:

```text
runs/UCMT_<backbone>_label_<labeled_percentage>/
```

For example:

```text
runs/UCMT_DeepLabv3p_label_0.05/
```

The output folder may contain:

```text
train_val.log
tensorboard*/
weights/
```

The `weights/` folder may contain checkpoint files such as:

```text
best.pth
last.pth
model1_best.pth
model2_best.pth
model3_best.pth
model1_last.pth
model2_last.pth
model3_last.pth
```

Different scripts may use different checkpoint names.

---

## 9. TensorBoard

To view training logs, run:

```bash
tensorboard --logdir ./runs
```

Then open:

```text
http://localhost:6006
```

---

## 10. Important Notes

### 10.1 GPU Setting

Several scripts contain:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

To use another GPU, change it manually:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

Or specify the GPU from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python train1_UADC.py
```

### 10.2 Configuration File Paths

Some scripts may contain local absolute paths, such as:

```text
/home/li/桌面/UCMT-main/configs/pascal/segformerb2_4x4.yaml
```

Replace them with paths that exist in the current folder, for example:

```bash
--config ./configs/pascal/segformerb2_4x4.yaml
--config2 ./configs/pascal/r50_dct_4x4.yaml
--config3 ./configs/pascal/r50_4x4.yaml
```

### 10.3 Dataset Class Matching

Different scripts use different dataset classes:

```text
ISICDataset      -> data/dataset.py
LADataset        -> data/dataset_3d.py
Polyp loader     -> data/split.py or related loader
SemiDataset      -> dataset/semi.py
SemiDatasetDCT   -> dataset/semi_dct.py
```

Make sure that the dataset path, file names, and image formats match the corresponding Dataset implementation.

### 10.4 Labeled Data Percentage

The parameter `labeled_percentage` controls how much labeled data is used.

For example:

```bash
--labeled_percentage 0.05
```

means that 5% of the training data is used as labeled data, while the remaining data is treated as unlabeled data.

---

## 11. Example Experiments

### UDAC with 5% Labels

```bash
python train1_UADC.py   --data_path ./dataset/ISIC/   --labeled_percentage 0.05   --num_epochs 25   --batch_size 4   --learning_rate 5e-4
```

### UDAC with 10% Labels

```bash
python train1_UADC.py   --data_path ./dataset/ISIC/   --labeled_percentage 0.1   --num_epochs 25   --batch_size 4   --learning_rate 5e-4
```

### Kvasir-CVC with 15% Labels

```bash
python train_kvasir_cvc.py   --image_root ./dataset/TrainDataset_cvc_kvasir/image/   --gt_root ./dataset/TrainDataset_cvc_kvasir/mask/   --labeled_percentage 0.15   --num_epochs 25   --batch_size 4   --config ./configs/pascal/segformerb2_4x4.yaml   --config2 ./configs/pascal/r50_dct_4x4.yaml   --config3 ./configs/pascal/r50_4x4.yaml
```

### 3D LA with 10% Labels

```bash
python train_3d.py   --data_path ./dataset/LA/   --labeled_percentage 0.1   --num_epochs 1000   --batch_size 4
```

---

## 12. FAQ

### Q1: What should I do if the dataset cannot be found?

Check the following:

- Whether `--data_path` is correct;
- Whether `--image_root` points to the image folder;
- Whether `--gt_root` points to the mask folder;
- Whether image and mask file names match;
- Whether the folder names match the Dataset class.

### Q2: What should I do if the configuration file cannot be found?

Check whether `--config`, `--config2`, and `--config3` point to existing YAML files in the current folder.

### Q3: What should I do if GPU memory is insufficient?

Reduce the batch size:

```bash
--batch_size 2
```

Or reduce the input image size:

```bash
--image_size 224
```

For 3D tasks, reduce the input volume size:

```bash
--image_size [64,96,96]
```

### Q4: How can I change the GPU?

Use:

```bash
CUDA_VISIBLE_DEVICES=1 python train1_UADC.py
```

or modify the following line in the script:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### Q5: How can I resume training?

The current scripts mainly train from scratch and save checkpoints. To resume training, add checkpoint loading code, for example:

```python
checkpoint = torch.load("./runs/xxx/weights/last.pth")
model.load_state_dict(checkpoint)
```

Then continue training.

---

## 13. Citation / Acknowledgement

If this project is used in academic experiments, please clearly describe the UDAC semi-supervised medical image segmentation framework, the dataset used, the labeled data percentage, the network architecture, and the training configuration.

---

## 14. TODO

Future improvements may include:

- Add a unified `requirements.txt`;
- Add a unified configuration file;
- Merge duplicate training scripts;
- Add a testing script;
- Add an inference script;
- Add dataset splitting scripts;
- Add automatic metric saving to CSV files;
- Rename `train_piture.py` to `train_picture.py`.
