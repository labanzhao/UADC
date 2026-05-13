# UCMT Medical Image Segmentation Training Code

本项目包含多组基于 PyTorch 的医学图像分割训练脚本，主要用于半监督语义分割实验。代码支持 2D 医学图像分割数据集（如 ISIC、Kvasir-CVC）和 3D 医学图像分割数据集（如 LA 数据集），并包含 UCMT、Mean Teacher、DCT 分支、多模型协同训练等实验版本。

> 注意：本 README 根据当前提供的 `train*.py` 训练脚本整理。由于部分路径、配置文件路径和数据路径在代码中为本地路径或占位路径，运行前需要根据自己的环境进行修改。

---

## 1. 项目功能简介

本项目主要实现以下功能：

- 医学图像二分类分割训练；
- 支持半监督学习场景，将训练集划分为有标签数据和无标签数据；
- 支持 2D 图像分割任务，例如 ISIC 皮肤病灶分割、Kvasir-CVC 息肉分割；
- 支持 3D 图像分割任务，例如 LA 数据集；
- 支持 DeepLabv3+、UNet、VNet 等分割网络；
- 支持 Mean Teacher / UCMT / 多模型协同训练框架；
- 支持 CutMix / BoxMask 数据增强；
- 支持 DCT 频域增强或 DCT 分支训练；
- 支持 TensorBoard 日志记录和模型权重保存。

---

## 2. 代码文件说明

| 文件名 | 主要用途 |
|---|---|
| `train.py` | 2D ISIC 半监督分割训练主脚本，包含多模型 / DCT / UCMT 相关训练逻辑。 |
| `train1.py` | `train.py` 的实验变体，主要用于 2D ISIC 类数据训练。 |
| `train_ucmt_isic.py` | 面向 ISIC 数据集的 UCMT 训练脚本。 |
| `train_UCMT.py` | 面向 Kvasir-CVC / 息肉数据的 UCMT 训练脚本，使用 `image_root` 和 `gt_root`。 |
| `train_kvasir_cvc.py` | Kvasir-CVC 息肉图像分割训练脚本，包含 DCT / 多模型配置。 |
| `train1_kvasir_cvc.py` | Kvasir-CVC 训练的另一实验版本。 |
| `train1_UADC.py` | UADC 相关训练实验脚本。 |
| `train_meanteacher.py` | Mean Teacher 半监督训练脚本。 |
| `train_3d.py` | 3D 医学图像半监督分割训练脚本，默认使用 VNet。 |
| `train_3d_6model.py` | 3D 多模型训练实验脚本。 |
| `train_piture.py` | 图像实验 / 可视化相关训练变体，文件名可能应为 `train_picture.py`。 |

---

## 3. 环境依赖

建议使用 Linux + NVIDIA GPU 环境运行。

### 3.1 基础环境

```bash
python >= 3.8
CUDA >= 11.x
PyTorch >= 1.10
```

### 3.2 Python 依赖

根据代码中的 import，建议安装以下依赖：

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm easydict pyyaml opencv-python tensorboard
pip install unfoldNd
```

如果使用分布式训练或特定模型结构，还需要保证项目中的以下模块存在：

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

## 4. 项目目录建议

建议将项目组织为如下结构：

```text
UCMT-main/
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
├── train_ucmt_isic.py
├── train_kvasir_cvc.py
├── train_meanteacher.py
├── train_3d.py
└── README.md
```

---

## 5. 数据准备

### 5.1 ISIC 数据集格式

适用于以下脚本：

```text
train.py
train1.py
train_ucmt_isic.py
train_meanteacher.py
train_piture.py
train1_UADC.py
```

这些脚本通常使用参数：

```bash
--data_path YOUR_DATA_PATH
```

建议数据目录结构如下：

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

具体目录名称需要与 `data/dataset.py` 中的 `ISICDataset` 实现保持一致。

### 5.2 Kvasir-CVC 数据集格式

适用于以下脚本：

```text
train_kvasir_cvc.py
train1_kvasir_cvc.py
train_UCMT.py
```

这些脚本通常使用参数：

```bash
--image_root PATH_TO_IMAGES
--gt_root PATH_TO_MASKS
```

建议目录结构如下：

```text
TrainDataset_cvc_kvasir/
├── image/
│   ├── xxx.png
│   └── ...
└── mask/
    ├── xxx.png
    └── ...
```

示例：

```bash
python train_kvasir_cvc.py \
  --image_root ./dataset/TrainDataset_cvc_kvasir/image/ \
  --gt_root ./dataset/TrainDataset_cvc_kvasir/mask/ \
  --labeled_percentage 0.15
```

### 5.3 3D LA 数据集格式

适用于：

```text
train_3d.py
train_3d_6model.py
```

这些脚本使用：

```bash
--data_path YOUR_3D_DATA_PATH
```

默认 3D 输入尺寸为：

```text
[80, 112, 112]
```

对应参数：

```bash
--image_size [80,112,112]
```

具体数据读取方式由 `data/dataset_3d.py` 中的 `LADataset` 决定。

---

## 6. 快速开始

### 6.1 训练 ISIC 数据集

```bash
python train_ucmt_isic.py \
  --data_path ./dataset/ISIC/ \
  --backbone DeepLabv3p \
  --image_size 256 \
  --labeled_percentage 0.1 \
  --num_epochs 25 \
  --batch_size 4 \
  --learning_rate 1e-4
```

或者使用主训练脚本：

```bash
python train.py \
  --data_path ./dataset/ISIC/ \
  --config ./configs/pascal/segformerb2_4x4.yaml \
  --config2 ./configs/pascal/r50_dct_4x4.yaml \
  --config3 ./configs/pascal/r50_4x4.yaml
```

### 6.2 训练 Kvasir-CVC 数据集

```bash
python train_kvasir_cvc.py \
  --image_root ./dataset/TrainDataset_cvc_kvasir/image/ \
  --gt_root ./dataset/TrainDataset_cvc_kvasir/mask/ \
  --backbone DeepLabv3p \
  --image_size 256 \
  --labeled_percentage 0.15 \
  --num_epochs 25 \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --config ./configs/pascal/segformerb2_4x4.yaml \
  --config2 ./configs/pascal/r50_dct_4x4.yaml \
  --config3 ./configs/pascal/r50_4x4.yaml
```

### 6.3 训练 3D LA 数据集

```bash
python train_3d.py \
  --data_path ./dataset/LA/ \
  --backbone VNet \
  --labeled_percentage 0.1 \
  --num_epochs 1000 \
  --batch_size 4 \
  --learning_rate 1e-4
```

### 6.4 运行 Mean Teacher

```bash
python train_meanteacher.py \
  --data_path ./dataset/ISIC/ \
  --backbone DeepLabv3p \
  --labeled_percentage 0.05 \
  --num_epochs 25 \
  --batch_size 4
```

---

## 7. 常用参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--seed` | `1` | 随机种子。 |
| `--project` | `./runs/UCMT` | 实验结果保存路径。 |
| `--backbone` | `DeepLabv3p` / `VNet` | 分割网络骨干。2D 脚本多为 `DeepLabv3p`，3D 脚本为 `VNet`。 |
| `--data_path` | `YOUR_DATA_PATH` | ISIC 或 3D 数据集路径。 |
| `--image_root` | `YOUR_DATA_PATH` | Kvasir-CVC 图像路径。 |
| `--gt_root` | `YOUR_DATA_PATH` | Kvasir-CVC 标签路径。 |
| `--image_size` | `256` 或 `[80,112,112]` | 输入图像大小。2D 为单个整数，3D 为三维尺寸。 |
| `--labeled_percentage` | `0.05` / `0.1` / `0.15` / `0.3` | 有标签数据比例。 |
| `--is_cutmix` | `False` | 是否启用 CutMix / BoxMask。 |
| `--mix_prob` | `0.5` | 幅度混合或增强概率。 |
| `--topk` | `1` 或 `2` | 选择 top-k 预测或不确定性区域时使用。 |
| `--num_epochs` | `25` / `50` / `1000` | 训练轮数。 |
| `--batch_size` | `4` | 批大小。 |
| `--num_workers` | `2` | DataLoader 线程数。 |
| `--in_channels` | `3` 或 `1` | 输入通道数。2D RGB 为 3，3D 灰度为 1。 |
| `--num_classes` | `2` | 分类类别数，默认二分类分割。 |
| `--pretrained` | `True` | 是否使用预训练权重。 |
| `--learning_rate` | `1e-4` / `5e-4` | 学习率。 |
| `--config` | YAML 路径 | 第一组模型配置文件。 |
| `--config2` | YAML 路径 | DCT 或第二模型配置文件。 |
| `--config3` | YAML 路径 | 第三模型配置文件。 |
| `--thr` | `0.95` | 伪标签置信度阈值。 |
| `--uw` | `1.0` | 无监督损失权重。 |
| `--amp` | `False` | 是否启用自动混合精度训练。 |

---

## 8. 训练流程说明

整体训练流程如下：

1. 读取命令行参数；
2. 设置随机种子；
3. 创建实验保存目录；
4. 加载训练集和验证集；
5. 按 `labeled_percentage` 划分有标签数据和无标签数据；
6. 将有标签数据重复采样，使其长度接近或等于训练集长度；
7. 构建 DataLoader；
8. 初始化分割模型；
9. 定义 Dice / DSC 损失函数；
10. 定义优化器，例如 AdamW 或 SGD；
11. 进行监督损失、无监督一致性损失、伪标签损失或 DCT 分支损失训练；
12. 使用 TensorBoard 和日志文件记录训练过程；
13. 保存 `best.pth`、`last.pth` 或多模型权重。

---

## 9. 输出结果

训练完成后，结果默认保存在：

```text
runs/UCMT_<backbone>_label_<labeled_percentage>/
```

目录中通常包含：

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

不同脚本保存的权重名称可能不同，例如：

- `best.pth`
- `last.pth`
- `model1_best.pth`
- `model2_best.pth`
- `model3_best.pth`
- `model1_last.pth`
- `model2_last.pth`
- `model3_last.pth`

---

## 10. 使用 TensorBoard 查看日志

训练时脚本会创建 TensorBoard 日志目录：

```text
runs/UCMT_xxx/tensorboardxxx/
```

查看方式：

```bash
tensorboard --logdir ./runs
```

然后在浏览器中打开：

```text
http://localhost:6006
```

---

## 11. 注意事项

### 11.1 修改 GPU 设置

多个脚本中固定设置了：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

如果需要使用其他 GPU，请修改为：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

或者删除该行，在命令行中指定：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

### 11.2 修改配置文件路径

部分脚本中默认配置文件路径为本地绝对路径，例如：

```text
/home/li/桌面/UCMT-main/configs/pascal/segformerb2_4x4.yaml
```

运行前应改为当前项目中的相对路径：

```bash
--config ./configs/pascal/segformerb2_4x4.yaml
--config2 ./configs/pascal/r50_dct_4x4.yaml
--config3 ./configs/pascal/r50_4x4.yaml
```

### 11.3 数据集类需要对应

不同脚本调用的数据集类不同：

```text
ISICDataset      -> data/dataset.py
LADataset        -> data/dataset_3d.py
Polyp loader     -> data/split.py 或相关 loader
SemiDataset      -> dataset/semi.py
SemiDatasetDCT   -> dataset/semi_dct.py
```

请确保数据路径、文件名、图像格式与对应 Dataset 类中的读取逻辑一致。

### 11.4 有标签比例

`labeled_percentage` 控制半监督训练中有标签数据的比例。例如：

```bash
--labeled_percentage 0.05
```

表示只使用 5% 训练数据作为有标签样本，其余样本用于无标签训练。

---

## 12. 示例实验命令

### ISIC 5% 标签实验

```bash
python train.py \
  --data_path ./dataset/ISIC/ \
  --labeled_percentage 0.05 \
  --num_epochs 25 \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --config ./configs/pascal/segformerb2_4x4.yaml \
  --config2 ./configs/pascal/r50_dct_4x4.yaml \
  --config3 ./configs/pascal/r50_4x4.yaml
```

### ISIC 10% 标签实验

```bash
python train_ucmt_isic.py \
  --data_path ./dataset/ISIC/ \
  --labeled_percentage 0.1 \
  --num_epochs 25 \
  --batch_size 4
```

### Kvasir-CVC 15% 标签实验

```bash
python train_kvasir_cvc.py \
  --image_root ./dataset/TrainDataset_cvc_kvasir/image/ \
  --gt_root ./dataset/TrainDataset_cvc_kvasir/mask/ \
  --labeled_percentage 0.15 \
  --num_epochs 25 \
  --batch_size 4 \
  --config ./configs/pascal/segformerb2_4x4.yaml \
  --config2 ./configs/pascal/r50_dct_4x4.yaml \
  --config3 ./configs/pascal/r50_4x4.yaml
```

### 3D LA 10% 标签实验

```bash
python train_3d.py \
  --data_path ./dataset/LA/ \
  --labeled_percentage 0.1 \
  --num_epochs 1000 \
  --batch_size 4
```

---

## 13. 常见问题

### Q1: 报错找不到配置文件怎么办？

请检查 `--config`、`--config2`、`--config3` 是否为当前机器上的真实路径。建议使用相对路径，不要使用原作者机器上的绝对路径。

### Q2: 报错找不到数据怎么办？

请检查：

- `--data_path` 是否正确；
- `--image_root` 是否指向图像文件夹；
- `--gt_root` 是否指向标签文件夹；
- 图像和标签文件名是否一一对应；
- Dataset 类中要求的目录名称是否与实际目录一致。

### Q3: 显存不足怎么办？

可以尝试：

```bash
--batch_size 2
```

或者减小输入尺寸：

```bash
--image_size 224
```

3D 任务也可以减小：

```bash
--image_size [64,96,96]
```

### Q4: 如何更换 GPU？

命令行指定：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py
```

或修改脚本中的：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### Q5: 如何继续训练？

当前脚本中主要是从头训练并保存权重。如果需要断点续训，可以在脚本中添加：

```python
checkpoint = torch.load("./runs/xxx/weights/last.pth")
model.load_state_dict(checkpoint)
```

然后继续执行训练。

---

## 14. Citation / Acknowledgement

如果本项目用于论文实验，请在论文或报告中说明使用了半监督医学图像分割框架，并注明所使用的数据集、标签比例、网络结构和训练配置。

---

## 15. TODO

后续可以进一步完善：

- 添加统一的 `requirements.txt`；
- 添加统一的 `config.yaml`；
- 整合多个重复训练脚本；
- 增加测试脚本 `test.py`；
- 增加模型推理脚本 `inference.py`；
- 增加数据集划分脚本；
- 增加自动保存训练指标为 CSV 的功能；
- 修正 `train_piture.py` 文件名为 `train_picture.py`。

