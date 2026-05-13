# UCMT Medical Image Segmentation Training Code

This project contains PyTorch-based training scripts for semi-supervised medical image segmentation. It supports 2D medical image segmentation datasets such as ISIC and Kvasir-CVC, as well as 3D medical image segmentation datasets such as LA. The code includes multiple experimental settings, including UCMT, Mean Teacher, DCT-based training, and multi-model collaborative learning.

> This README should be placed in the project root directory as:
>
> ```text
> UCMT-main/README.md
> ```

---

## 1. Project Overview

The project supports:

- Binary medical image segmentation;
- Semi-supervised learning with labeled and unlabeled data;
- 2D segmentation tasks, including ISIC skin lesion segmentation and Kvasir-CVC polyp segmentation;
- 3D segmentation tasks, including LA dataset segmentation;
- Segmentation backbones such as DeepLabv3+, UNet, and VNet;
- UCMT, Mean Teacher, and multi-model collaborative training;
- CutMix / BoxMask data augmentation;
- DCT-based frequency-domain enhancement or DCT branch training;
- TensorBoard logging and model checkpoint saving.

---

## 2. Recommended Project Structure

Please place this README in the project root directory:

```text
UCMT-main/
├── README.md
├── configs/
│   └── pascal/
│       ├── segformerb2_4x4.yaml
│       ├── r50_dct_4x4.yaml
│       └── r50_4x4.yaml
├── data/
│   ├── dataset.py
│   ├── dataset_3d.py
│   └── split.py
├── dataset/
│   ├── semi.py
│   ├── semi_dct.py
│   └── dct_transform.py
├── model/
│   └── semseg/
├── models/
│   └── deeplabv3.py
├── util/
├── utils/
├── train.py
├── train1.py
├── train_ucmt_isic.py
├── train_UCMT.py
├── train_kvasir_cvc.py
├── train1_kvasir_cvc.py
├── train1_UADC.py
├── train_meanteacher.py
├── train_3d.py
├── train_3d_6model.py
└── train_piture.py
```

---

## 3. File Description

| File | Description |
|---|---|
| `train.py` | Main 2D ISIC semi-supervised segmentation training script. It contains UCMT, DCT, and multi-model related training logic. |
| `train1.py` | Experimental variant of `train.py`, mainly used for 2D ISIC-like datasets. |
| `train_ucmt_isic.py` | UCMT training script for the ISIC dataset. |
| `train_UCMT.py` | UCMT training script for Kvasir-CVC / polyp segmentation datasets. |
| `train_kvasir_cvc.py` | Kvasir-CVC polyp segmentation training script with DCT and multi-model configurations. |
| `train1_kvasir_cvc.py` | Another experimental training script for Kvasir-CVC. |
| `train1_UADC.py` | Training script for UADC-related experiments. |
| `train_meanteacher.py` | Mean Teacher semi-supervised training script. |
| `train_3d.py` | 3D medical image segmentation training script using VNet by default. |
| `train_3d_6model.py` | 3D multi-model collaborative training script. |
| `train_piture.py` | Image experiment or visualization-related training variant. The intended name may be `train_picture.py`. |

---

## 4. Environment Requirements

A Linux system with an NVIDIA GPU is recommended.

### 4.1 Basic Environment

```bash
python >= 3.8
CUDA >= 11.x
PyTorch >= 1.10
```

### 4.2 Python Dependencies

Install the main dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm easydict pyyaml opencv-python tensorboard
pip install unfoldNd
```

The following project modules should also exist:

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

## 5. Dataset Preparation

### 5.1 ISIC Dataset

Applicable scripts:

```text
train.py
train1.py
train_ucmt_isic.py
train_meanteacher.py
train_piture.py
train1_UADC.py
```

These scripts usually use:

```bash
--data_path YOUR_DATA_PATH
```

Recommended dataset structure:

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

The actual folder names must match the implementation of `ISICDataset` in `data/dataset.py`.

### 5.2 Kvasir-CVC Dataset

Applicable scripts:

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

Recommended dataset structure:

```text
TrainDataset_cvc_kvasir/
├── image/
│   ├── xxx.png
│   └── ...
└── mask/
    ├── xxx.png
    └── ...
```

### 5.3 3D LA Dataset

Applicable scripts:

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

The exact loading process depends on `LADataset` in `data/dataset_3d.py`.

---

## 6. Quick Start

### 6.1 Train on ISIC

```bash
python train_ucmt_isic.py   --data_path ./dataset/ISIC/   --backbone DeepLabv3p   --image_size 256   --labeled_percentage 0.1   --num_epochs 25   --batch_size 4   --learning_rate 1e-4
```

Or use the main training script:

```bash
python train.py   --data_path ./dataset/ISIC/   --config ./configs/pascal/segformerb2_4x4.yaml   --config2 ./configs/pascal/r50_dct_4x4.yaml   --config3 ./configs/pascal/r50_4x4.yaml
```

### 6.2 Train on Kvasir-CVC

```bash
python train_kvasir_cvc.py   --image_root ./dataset/TrainDataset_cvc_kvasir/image/   --gt_root ./dataset/TrainDataset_cvc_kvasir/mask/   --backbone DeepLabv3p   --image_size 256   --labeled_percentage 0.15   --num_epochs 25   --batch_size 4   --learning_rate 5e-4   --config ./configs/pascal/segformerb2_4x4.yaml   --config2 ./configs/pascal/r50_dct_4x4.yaml   --config3 ./configs/pascal/r50_4x4.yaml
```

### 6.3 Train on 3D LA

```bash
python train_3d.py   --data_path ./dataset/LA/   --backbone VNet   --labeled_percentage 0.1   --num_epochs 1000   --batch_size 4   --learning_rate 1e-4
```

### 6.4 Run Mean Teacher Training

```bash
python train_meanteacher.py   --data_path ./dataset/ISIC/   --backbone DeepLabv3p   --labeled_percentage 0.05   --num_epochs 25   --batch_size 4
```

---

## 7. Common Parameters

| Parameter | Default Value | Description |
|---|---:|---|
| `--seed` | `1` | Random seed. |
| `--project` | `./runs/UCMT` | Path for saving experiment results. |
| `--backbone` | `DeepLabv3p` / `VNet` | Segmentation backbone. |
| `--data_path` | `YOUR_DATA_PATH` | Path to the ISIC or 3D dataset. |
| `--image_root` | `YOUR_DATA_PATH` | Image path for Kvasir-CVC. |
| `--gt_root` | `YOUR_DATA_PATH` | Ground-truth mask path for Kvasir-CVC. |
| `--image_size` | `256` or `[80,112,112]` | Input image or volume size. |
| `--labeled_percentage` | `0.05`, `0.1`, `0.15`, `0.3` | Percentage of labeled data. |
| `--is_cutmix` | `False` | Whether to enable CutMix / BoxMask. |
| `--mix_prob` | `0.5` | Probability for amplitude mixing or augmentation. |
| `--topk` | `1` or `2` | Top-k selection setting. |
| `--num_epochs` | `25`, `50`, `1000` | Number of training epochs. |
| `--batch_size` | `4` | Batch size. |
| `--num_workers` | `2` | Number of DataLoader workers. |
| `--in_channels` | `3` or `1` | Input channels. 2D RGB images use 3 channels; 3D volumes usually use 1 channel. |
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

## 8. Training Pipeline

The general training pipeline is:

1. Parse command-line arguments;
2. Set the random seed;
3. Create experiment directories;
4. Load training and validation datasets;
5. Split the training data into labeled and unlabeled subsets according to `labeled_percentage`;
6. Repeat the labeled subset so that its length is close to or equal to the full training set;
7. Build DataLoaders;
8. Initialize segmentation models;
9. Define Dice / DSC loss functions;
10. Define optimizers such as AdamW or SGD;
11. Train with supervised loss, unsupervised consistency loss, pseudo-label loss, or DCT branch loss;
12. Record training logs using TensorBoard and log files;
13. Save model checkpoints.

---

## 9. Output Results

By default, training results are saved in:

```text
runs/UCMT_<backbone>_label_<labeled_percentage>/
```

The directory usually contains:

```text
runs/UCMT_DeepLabv3p_label_0.1/
├── train_val.log
├── tensorboardMayxx_xx-xx-xx/
└── weights/
    ├── best.pth
    ├── last.pth
    ├── model1_best.pth
    ├── model2_best.pth
    ├── model3_best.pth
    ├── model1_last.pth
    ├── model2_last.pth
    └── model3_last.pth
```

Different scripts may save checkpoints with different names, such as:

- `best.pth`
- `last.pth`
- `model1_best.pth`
- `model2_best.pth`
- `model3_best.pth`
- `model1_last.pth`
- `model2_last.pth`
- `model3_last.pth`

---

## 10. View Logs with TensorBoard

During training, the scripts create a TensorBoard log directory:

```text
runs/UCMT_xxx/tensorboardxxx/
```

Run:

```bash
tensorboard --logdir ./runs
```

Then open:

```text
http://localhost:6006
```

---

## 11. Important Notes

### 11.1 GPU Settings

Several scripts contain:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

To use another GPU, modify it, for example:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

Alternatively, remove that line and specify the GPU from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

### 11.2 Configuration File Paths

Some scripts contain local absolute paths, for example:

```text
/home/li/桌面/UCMT-main/configs/pascal/segformerb2_4x4.yaml
```

Replace them with relative paths:

```bash
--config ./configs/pascal/segformerb2_4x4.yaml
--config2 ./configs/pascal/r50_dct_4x4.yaml
--config3 ./configs/pascal/r50_4x4.yaml
```

### 11.3 Dataset Classes

Different scripts call different dataset classes:

```text
ISICDataset      -> data/dataset.py
LADataset        -> data/dataset_3d.py
Polyp loader     -> data/split.py or related loader
SemiDataset      -> dataset/semi.py
SemiDatasetDCT   -> dataset/semi_dct.py
```

Make sure dataset paths, file names, and image formats match the corresponding Dataset implementation.

### 11.4 Labeled Data Percentage

`labeled_percentage` controls the proportion of labeled samples used in semi-supervised training.

For example:

```bash
--labeled_percentage 0.05
```

This means that only 5% of the training data is used as labeled data, while the remaining data is used as unlabeled data.

---

## 12. Example Experiment Commands

### ISIC with 5% Labels

```bash
python train.py   --data_path ./dataset/ISIC/   --labeled_percentage 0.05   --num_epochs 25   --batch_size 4   --learning_rate 5e-4   --config ./configs/pascal/segformerb2_4x4.yaml   --config2 ./configs/pascal/r50_dct_4x4.yaml   --config3 ./configs/pascal/r50_4x4.yaml
```

### ISIC with 10% Labels

```bash
python train_ucmt_isic.py   --data_path ./dataset/ISIC/   --labeled_percentage 0.1   --num_epochs 25   --batch_size 4
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

## 13. FAQ

### Q1: What should I do if the configuration file cannot be found?

Check whether `--config`, `--config2`, and `--config3` point to valid paths on your machine. Relative paths are recommended.

### Q2: What should I do if the dataset cannot be found?

Check:

- Whether `--data_path` is correct;
- Whether `--image_root` points to the image folder;
- Whether `--gt_root` points to the mask folder;
- Whether image and mask file names correspond to each other;
- Whether the folder names match the requirements in the Dataset class.

### Q3: What should I do if GPU memory is insufficient?

Reduce the batch size:

```bash
--batch_size 2
```

Or reduce the input image size:

```bash
--image_size 224
```

For 3D tasks, reduce the volume size:

```bash
--image_size [64,96,96]
```

### Q4: How can I change the GPU?

Use:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py
```

Or modify:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### Q5: How can I resume training?

The current scripts mainly train from scratch and save checkpoints. To resume training, add code similar to:

```python
checkpoint = torch.load("./runs/xxx/weights/last.pth")
model.load_state_dict(checkpoint)
```

Then continue training.

---

## 14. Citation / Acknowledgement

If this project is used for paper experiments, please describe the semi-supervised medical image segmentation framework in your paper or report. Also specify the dataset, labeled data percentage, network architecture, and training configuration used in the experiments.

---

## 15. TODO

Future improvements may include:

- Add a unified `requirements.txt`;
- Add a unified `config.yaml`;
- Merge duplicate training scripts;
- Add a testing script `test.py`;
- Add an inference script `inference.py`;
- Add a dataset splitting script;
- Add automatic training metric saving as CSV files;
- Rename `train_piture.py` to `train_picture.py`.
