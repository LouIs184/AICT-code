# AICTVer2 项目文档

> 版本: 2.0.0  
> 基于: AICT (AI CT Image Reconstruction) 重构  
> 重构依据: `ARCHITECTURE_ANALYSIS.md`

---

## 目录

1. [项目概述](#1-项目概述)
2. [重构动机](#2-重构动机)
3. [架构总览](#3-架构总览)
4. [模块详解](#4-模块详解)
   - 4.1 [配置系统 — `src/config.py`](#41-配置系统---srcconfigpy)
   - 4.2 [模型层 — `src/models/`](#42-模型层---srcmodels)
   - 4.3 [损失函数 — `src/losses/`](#43-损失函数---srclosses)
   - 4.4 [数据加载 — `src/data/`](#44-数据加载---srcdata)
   - 4.5 [工具模块 — `src/utils/`](#45-工具模块---srcutils)
   - 4.6 [优化器 — `src/optimizers/`](#46-优化器---srcoptimizers)
   - 4.7 [训练引擎 — `src/trainer.py`](#47-训练引擎---srctrainerpy)
5. [实验配置指南](#5-实验配置指南)
6. [训练/推理流程](#6-训练推理流程)
7. [如何新增实验](#7-如何新增实验)
8. [从原版 AICT 迁移](#8-从原版-aict-迁移)
9. [技术债务对照](#9-技术债务对照)

---

## 1. 项目概述

**AICTVer2** 是一个基于深度学习的 CT 图像重建项目。它的核心任务是用神经网络对**低剂量 CT（LDCT）** 图像进行降噪和质量增强，服务于双能相位 CT 扫描仪产生的数据。

项目覆盖**两个领域**：

| 领域 | 输入 | 输出 | 数据形状 | 物理含义 |
|------|------|------|---------|---------|
| **PhaseCT (图像域)** | `rec_pbi_*.raw` | `rec_no_pbi.raw` | (160, 512, 512) | 重建后的 CT 切片体数据 |
| **ProjCT (投影域)** | `proj_pbi_*.raw` | `proj_no_pbi.raw` | (720, 128, 512) | 原始投影视角序列 |

核心技术栈：
- **模型架构**: DenseUNet2d（密集连接 U-Net，主要）、UNet2d（基线变体）
- **训练策略**: MixUp/CutMix 数据增强、WGAN-GP 对抗训练、VGG 感知损失、深度监督、SophiaG 二阶优化器
- **框架**: PyTorch 2.0+、TensorBoard、YAML 配置

---

## 2. 重构动机

原版 AICT 项目经过多次迭代积累了相当的技术债务，主要痛点如下：

### 🔴 严重: 9 个训练脚本大量重复

```
train_unet_2d.py
train_unet_proj_2d.py          # ← 90% 的代码与上面相同
train_unet_2d_mixup.py         # ← 同上
train_unet_2d_cutmix.py        # ← 同上
train_unet_2d_gan.py           # ← 同上
train_unet_2d_vgg.py           # ← 同上
train_unet_2d_vgg_mgpu.py      # ← 同上
train_unet_2d_mixup_deepsurperviser.py  # ← 同上
train_unet_2d_mixup_sophia.py  # ← 同上
```

每个脚本都包含一份完整的 `train()` + `valid()` + `if __name__` 三件套。差异仅在**模型选择、数据增强、损失权重、优化器类型**这几个维度上。结果是：
- 要改日志格式？改 9 个文件。
- 要加验证指标？改 9 个文件。
- 要修 bug？改 9 个文件。

**AICTVer2 方案**: 一个统一的 `Trainer` 类 + YAML 配置驱动行为差异。新增实验只需写几行 YAML，无需复制代码。

### 🟡 中等: 模块职责混乱

- `utils.py` 同时包含 SSIM（指标）、模型 I/O（持久化）、MixUp/CutMix（数据增强）、高斯滤波（图像处理）—— 4 个不相关的职责
- `pytools.py` 混合了 raw 数据读取、DICOM 读取、泊松噪声模拟、3D 可视化
- 根目录下 20+ 个文件，使用 `from models import *` 通配符导入

**AICTVer2 方案**: 每个模块一个文件，按职责组织到 `src/models/`、`src/losses/`、`src/data/`、`src/utils/`、`src/optimizers/` 子包中。

---

## 3. 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                        用户入口 (Entry Points)                     │
│                                                                  │
│  train.py  test.py  prepare_data.py  prepare_proj_data.py        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    配置系统 (src/config.py)                        │
│                                                                  │
│  YAML 文件 ──→ Config dataclass ──→ 注入 Trainer                  │
│                                                                  │
│  支持: 嵌套 dataclass / 部分覆盖 / CLI 参数覆盖                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    训练引擎 (src/trainer.py)                       │
│                                                                  │
│  Trainer                                                         │
│  ├── _build_dataloaders()     ← DataConfig                       │
│  ├── _build_model()           ← ModelConfig                      │
│  ├── _build_losses()          ← LossConfig                       │
│  ├── _build_optimizers()      ← OptimizerConfig + GANConfig      │
│  ├── _build_schedulers()      ← SchedulerConfig                  │
│  ├── _train_one_epoch()       ← 标准 / GAN / 深度监督            │
│  ├── _validate()              ← MSE + SSIM + PSNR                │
│  └── fit()                    ← 主循环                           │
└───────┬──────────────┬──────────────┬───────────────────────────┘
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  src/models/  │ │  src/losses/  │ │   src/data/   │
│  (网络结构)    │ │  (损失函数)    │ │  (数据加载)    │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ src/utils/    │ │src/optimizers│ │   YAML 配置      │
│ (工具函数)     │ │ (SophiaG)    │ │ experiments/*.yaml│
└──────────────┘ └──────────────┘ └──────────────────┘
```

---

## 4. 模块详解

### 4.1 配置系统 — `src/config.py`

配置是整个系统的"控制面板"。采用 **YAML + dataclass** 组合：

```yaml
# experiments/denseunet_phasect.yaml
experiment:
  name: DenseUNet_PhaseCT
  seed: 1234

data:
  reader: raw                    # raw | dicom | proj
  train_txt: ./train_img.txt
  batch_size: 12
  num_workers: 8

model:
  name: DenseUNet2d              # DenseUNet2d | UNet2d | UNet2dDeepSuperviser
  in_chl: 1
  out_chl: 1
  model_chl: 32

loss:
  mse_weight: 1.0                # 各损失分量权重
  charbonnier_weight: 0.0
  ssim_weight: 0.0
  vgg_weight: 0.0
  adv_weight: 0.0                # GAN 对抗损失权重
  deep_supervision: false

optimizer:
  name: Adam                     # Adam | AdamW | SophiaG
  lr: 0.0001
  betas: [0.9, 0.999]

scheduler:
  name: StepLR                   # StepLR | MultiStepLR | CosineAnnealingLR
  step_size: 2
  gamma: 0.95

training:
  epochs: 200
  save_dir: ./runs/DenseUNet_PhaseCT
  log_dir: ./runs/DenseUNet_PhaseCT
```

**设计要点**：
- 所有子配置都是 **dataclass**，类型安全，IDE 自动补全友好
- YAML 只写需要覆盖的字段即可——不存在的键被静默忽略
- 支持通过 CLI 覆盖：`--epochs 300 --lr 0.0005 --batch-size 24`
- `load_config()` 使用递归合并，YAML 中的嵌套键自动映射到对应 dataclass

### 4.2 模型层 — `src/models/`

```
src/models/
├── __init__.py          # 导出所有模型类
├── blocks.py            # 基础构建块
├── unet.py              # U-Net 全家桶
└── discriminator.py     # GAN 判别器
```

**`blocks.py`** — 积木式组件，所有 U-Net 变体共享：

| 类 | 用途 | 特点 |
|----|------|------|
| `ConvBnRelu2d` | 标准卷积块 | 可选的 BN 和 ReLU |
| `StackEncoder` | 标准编码器 | 2×Conv → MaxPool |
| `StackDecoder` | 标准解码器 | 上采样 → Concat → 2×Conv |
| `StackDenseEncoder2d` | 密集编码器 | 3 层 Conv 密集连接 + 残差 |
| `StackDenseBlock2d` | 密集瓶颈块 | 同上，无下采样 |
| `StackResDecoder2d` | 残差解码器 | 残差跳跃连接 |

**`unet.py`** — 三种 U-Net 变体：

```
┌─────────────────────────────────────────────┐
│  DenseUNet2d (主要模型)                      │
│                                              │
│  输入(1,512,512)                             │
│    ↓                                         │
│  begin: 32                                   │
│  down1: 32  [DenseEncode]                    │
│  down2: 64  [DenseEncode]                    │
│  down3: 128 [DenseEncode]                    │
│  down4: 256 [DenseEncode]                    │
│  center: 512 [DenseBlock]                    │
│  up4 → 256 [ResDecoder + Skip]               │
│  up3 → 128 [ResDecoder + Skip]               │
│  up2 → 64  [ResDecoder + Skip]               │
│  up1 → 32  [ResDecoder + Skip]               │
│  end: 1                                      │
│  output = ReLU(input + output)  ← 残差连接    │
└─────────────────────────────────────────────┘
```

```python
# 三种模型的使用场景
DenseUNet2d          → 主力模型，PhaseCT 和 ProjCT 都使用
UNet2d               → 基线/消融实验，CutMix/GAN/VGG 实验
UNet2dDeepSuperviser → 深度监督实验，返回多尺度中间输出
```

**`discriminator.py`** — GAN 判别器，WGAN-GP 训练用。

### 4.3 损失函数 — `src/losses/`

```
src/losses/
├── regularizers.py    # CharbonnierLoss, TVLoss
├── perceptual.py      # VGG 感知损失
└── ssim.py           # SSIM 指标
```

| 损失 | 类/函数 | 用途 | 典型权重 |
|------|---------|------|---------|
| MSE | `nn.MSELoss` | 基本像素级损失 | 1.0 |
| Charbonnier | `CharbonnierLoss` | L1 的可微近似，对大误差更鲁棒 | 0.0–1.0 |
| SSIM | `SSIM(as_loss=True)` | 结构相似性损失 | 0.0–0.1 |
| VGG Perceptual | `vgg_loss_calc` | 5 层 VGG 特征 MSE | 0.0–0.1 |
| TV | `TVLoss` | 总变分正则化，鼓励平滑 | 0.0–0.01 |
| GAN Adversarial | `-E[D(G(x))]` | 对抗损失（WGAN-GP） | 0.01 |

损失组合由配置中的权重控制。`Trainer._compute_loss()` 根据权重自动组合：

```python
# 等价于原版 train_unet_2d.py
loss:
  mse_weight: 1.0
  charbonnier_weight: 0.0   # 实际未使用，为 0

# 等价于 train_unet_2d_vgg.py （如果开启 vgg_weight）
loss:
  mse_weight: 1.0
  vgg_weight: 0.01
  ssim_weight: 0.0
```

### 4.4 数据加载 — `src/data/`

```
src/data/
├── raw_reader.py       # PhaseCT: 从 .raw 体数据按切片索引读取
├── dicom_reader.py     # DICOM: 从 .IMA 文件对读取
└── proj_reader.py      # ProjCT: 从 .raw 投影体数据按视角索引读取
```

所有 DataSet 返回统一的字典格式：

```python
{"ldct": torch.FloatTensor(1, H, W),   # 低剂量/含噪输入
 "rdct": torch.FloatTensor(1, H, W)}   # 常规剂量/参考输出
```

| 读取器 | 输入形状 | 输出形状 | 归一化方式 |
|--------|---------|---------|-----------|
| `raw_reader` | (160, 512, 512) | (1, 512, 512) | `(x + 1000) / 2000` 映射 HU [-1000, 1000] → [0, 1] |
| `dicom_reader` | 单个 DICOM | (1, H, W) | 原始像素值，仅 clip 负值 |
| `proj_reader` | (720, 128, 512) | (1, 128, 512) | `clip(x, 0, 0.28) / 0.28` 映射 [0, 0.28] → [0, 1] |

**关键优化**: `proj_reader` 使用文件 seek（`offset` 参数），只读取目标视角而非整个体数据，大幅提升 I/O 效率。

### 4.5 工具模块 — `src/utils/`

```
src/utils/
├── io_utils.py    # 模型保存/加载 (load_model, save_model, find_lastest_file)
├── metrics.py     # AverageMeter, normalization, PSNR 计算
├── augments.py    # MixUp_AUG, CutMix_AUG, CutMix_AUG_V2
└── viz.py         # gaussian_smooth, extract_patches_online
```

**职责拆分**（对比原版 `utils.py` 的改进）：

| 原版函数 | 原位置 | 新位置 |
|---------|--------|--------|
| `load_model` 系列 | `utils.py` | `io_utils.py` |
| `save_model` | `utils.py` | `io_utils.py` |
| `SSIM` 类 + `ssim()` | `utils.py` | `losses/ssim.py` |
| `AverageMeter` | `utils.py` | `metrics.py` |
| `normalization` | `utils.py` | `metrics.py` |
| `MixUp_AUG` | `utils.py` | `augments.py` |
| `CutMix_AUG` | `utils.py` | `augments.py` |
| `gaussian_smooth` | `utils.py` | `viz.py` |
| `extract_patches_online` | `utils.py` | `viz.py` |

### 4.6 优化器 — `src/optimizers/`

```
src/optimizers/
├── __init__.py
└── sophia.py          # SophiaG 优化器
```

SophiaG 是一种**二阶优化器**，使用对角 Hessian 估计进行梯度预处理。相比 Adam，它在 Transformer 和大模型训练中表现更优。

配置方式：

```yaml
optimizer:
  name: SophiaG
  lr: 0.0001
  betas: [0.965, 0.99]    # SophiaG 特有的 beta 参数
  weight_decay: 0.1       # 建议使用更大的 weight_decay
```

### 4.7 训练引擎 — `src/trainer.py`

这是重构的核心产出，用一个类替代了原来的 9 个训练脚本。

**初始化流程**：

```
Trainer.__init__()
  ├── _build_dataloaders()   → 根据 data.reader 选择 raw/dicom/proj
  ├── _build_model()         → 根据 model.name 实例化模型
  │                            (自动 DataParallel 包装)
  ├── _build_losses()        → 创建 MSE / Charbonnier / SSIM / TV
  ├── _build_optimizers()    → 根据 optimizer.name 选择 Adam/AdamW/SophiaG
  │                            (GAN 时额外创建 D 优化器)
  ├── _build_schedulers()    → StepLR / MultiStepLR / CosineAnnealingLR
  └── _build_writer()        → TensorBoard SummaryWriter
```

**训练循环**：

```
Trainer.fit()
  └── for epoch in range(epochs):
        ├── _train_one_epoch()
        │     ├── 数据增强 (MixUp / CutMix, 如果配置)
        │     ├── _train_standard_step()    ← 标准/深度监督
        │     │     ├── model(ldct) → predict
        │     │     ├── _compute_loss(predict, rdct)
        │     │     ├── optimizer.backward + step
        │     │     └── TensorBoard 日志
        │     └── _train_gan_step()         ← GAN 训练
        │           ├── D 更新 (真实 + 生成 + 梯度惩罚)
        │           ├── G 更新 (重建损失 + 对抗损失)
        │           └── TensorBoard 日志
        │
        ├── _validate()
        │     ├── model.eval()
        │     ├── MSE / SSIM / PSNR 统计
        │     └── TensorBoard 日志
        │
        └── save_model()          ← 检查点保存
```

**GAN 训练的特殊处理**：

原版 GAN 训练脚本（`train_unet_2d_gan.py`）的独特之处在于使用 `extract_patches_online` 将图像分成 4×4 的小块输入判别器。AICTVer2 的 `_train_gan_step` 保留了这一设计。

**深度监督的特殊处理**：

`UNet2dDeepSuperviser` 返回 5 个输出 `(conv5, up4, up3, up2, final)`，训练引擎自动检测元组返回值，对中间输出用 `F.interpolate` 调整目标尺寸后计算多尺度 MSE 损失。

---

## 5. 实验配置指南

### 5.1 配置文件是"增量覆盖"的

你不必写全所有字段。缺失的字段自动使用 dataclass 的默认值。

```yaml
# 只写你想改的
experiment:
  name: MyExperiment

data:
  train_txt: ./my_train.txt
  valid_txt: ./my_valid.txt
```

等价于所有其他字段取默认值（`batch_size: 12`, `model: DenseUNet2d`, `optimizer: Adam`, `epochs: 200` 等）。

### 5.2 已提供的实验模板

| 配置文件 | 对应原版 | 特点 |
|---------|---------|------|
| `denseunet_phasect.yaml` | `train_unet_2d.py` | PhaseCT 基线, DenseUNet2d, Adam, MSE |
| `denseunet_projct.yaml` | `train_unet_proj_2d.py` | ProjCT 基线, batch_size=36 |
| `denseunet_mixup.yaml` | `train_unet_2d_mixup.py` | MixUp 增强, AdamW |
| `unet_cutmix.yaml` | `train_unet_2d_cutmix.py` | CutMix V2, UNet2d |
| `unet_gan.yaml` | `train_unet_2d_gan.py` | WGAN-GP, D+G 优化器 |
| `unet_vgg.yaml` | `train_unet_2d_vgg.py` | VGG 感知损失 |
| `unet_deepsup.yaml` | `train_unet_2d_mixup_deepsurperviser.py` | 深度监督 + MixUp |
| `denseunet_sophia.yaml` | `train_unet_2d_mixup_sophia.py` | SophiaG 优化器 + MixUp |

---

## 6. 训练/推理流程

### 6.1 数据准备

```bash
# PhaseCT: 生成 train_img.txt / valid_img.txt
python prepare_data.py --data-dir /path/to/UIH_data --output ./

# ProjCT: 生成 train_proj_img.txt / valid_proj_img.txt
python prepare_proj_data.py --data-dir /path/to/UIH_data --output ./
```

生成的索引文件格式：

```
# train_img.txt (每行)
/path/to/rec_pbi_xxx.raw /path/to/rec_no_pbi.raw 42
/path/to/rec_pbi_xxx.raw /path/to/rec_no_pbi.raw 43
...
```

### 6.2 训练

```bash
# 基本用法
python train.py --config experiments/denseunet_phasect.yaml

# 恢复训练
python train.py --config experiments/denseunet_phasect.yaml --resume

# 覆盖参数
python train.py --config experiments/denseunet_phasect.yaml \
    --epochs 300 --lr 0.0005 --batch-size 24 --gpu "0,1"

# GAN 训练
python train.py --config experiments/unet_gan.yaml --gpu "1"
```

### 6.3 监控

```bash
tensorboard --logdir ./runs/DenseUNet_PhaseCT/logs
```

TensorBoard 记录：
- **标量**: `loss/train_mse`, `loss/valid_mse`, `loss/valid_ssim`, `valid/psnr`
- **图像**: `train img/ldct`, `train img/reference`, `train img/predict`, `train img/residual`
- **验证图像**: 同上, 前缀 `valid`

### 6.4 推理

```bash
# PhaseCT 推理
python test.py -c experiments/denseunet_phasect.yaml \
    --checkpoint ./runs/DenseUNet_PhaseCT/checkpoints/model_at_epoch_200.dat \
    --data-dir /path/to/data --output ./results/

# ProjCT 推理
python test.py -c experiments/denseunet_projct.yaml \
    --checkpoint ./runs/DenseUNet_ProjCT/checkpoints/model_at_epoch_200.dat \
    --data-dir /path/to/data --output ./results_proj/ --domain proj
```

推理脚本会自动：
1. 加载体数据（160 切片或 720 视角）
2. 逐切片/视角推理（跳过边界切片）
3. 反归一化回原始 HU/投影值
4. 保存 `.raw` 文件
5. 计算 PSNR 和 SSIM（PhaseCT 域）

---

## 7. 如何新增实验

### 场景一: 调参

已有配置，只需改参数：

```bash
python train.py -c experiments/denseunet_phasect.yaml \
    --epochs 300 --lr 0.0005 --batch-size 24
```

### 场景二: 新损失组合

创建一个新的 YAML 文件：

```yaml
# experiments/my_new_loss.yaml
experiment:
  name: DenseUNet_MSE_Charbonnier_SSIM

loss:
  mse_weight: 1.0
  charbonnier_weight: 0.1
  ssim_weight: 0.01

# 其余字段继承默认值，或用 include 方式（见场景三）
```

### 场景三: 基于现有配置的变体

复制一份现有配置，只改差异部分：

```bash
cp experiments/denseunet_phasect.yaml experiments/my_variant.yaml
# 编辑 my_variant.yaml ...
python train.py -c experiments/my_variant.yaml
```

### 场景四: 使用现有代码中没有的增强方法

如果新增的数据增强不在 `src/utils/augments.py` 中，按以下步骤：

1. 在 `augments.py` 中实现新类，要求 `aug(rgb_gt, rgb_noisy)` 接口
2. 在 `src/utils/__init__.py` 中导出
3. 在 `src/trainer.py` 的 `_build_model()` 后注册（在 `__init__` 的增强初始化部分添加分支）
4. 在 YAML 中设置 `augment.name` 为新名称

### 场景五: 新模型架构

1. 在 `src/models/` 中实现新模型类
2. 在 `src/models/__init__.py` 中导出
3. 在 `src/trainer.py` 的 `MODEL_MAP` 中注册
4. 在 YAML 中设置 `model.name` 为新类名

---

## 8. 从原版 AICT 迁移

### 检查点兼容

原版 AICT 的检查点可以直接加载，因为模型架构和 `state_dict` 键名完全一致：

```python
from src.models import DenseUNet2d
from src.utils.io_utils import load_model

model = DenseUNet2d()
checkpoint = torch.load("path/to/old_checkpoint.dat")
load_model(model, checkpoint)  # 直接兼容
```

### 数据索引迁移

原版的 `train_img.txt` 包含 AutoDL 服务器的绝对路径。需要用新的数据路径重新生成：

```bash
python prepare_data.py --data-dir /your/local/path --output ./
```

### 配置映射

```yaml
# 原版 train_unet_2d.py 中的硬编码值 → 对应 YAML 配置项

# 模型: DenseUNet2d() → model.name: DenseUNet2d
# 损失: MSELoss() + CharbonnierLoss() 但只用了 MSE
#       → loss.mse_weight: 1.0, loss.charbonnier_weight: 0.0
# 优化器: Adam(lr=0.0001) → optimizer.name: Adam, optimizer.lr: 0.0001
# 调度器: StepLR(step_size=2, gamma=0.95)
#       → scheduler.name: StepLR, scheduler.step_size: 2, scheduler.gamma: 0.95
# 轮数: 200 → training.epochs: 200
# 批量大小: 12 → data.batch_size: 12
# 保存路径: ./runs/DenseUNet_PhaseCT/checkpoints/ → training.save_dir
```

---

## 9. 技术债务对照

| 原版问题 | 严重程度 | 解决方案 | 状态 |
|---------|---------|---------|------|
| 9 个训练脚本重复 | 🔴 严重 | 统一 Trainer + YAML 配置 | ✅ 已解决 |
| 根目录 20+ 文件 | 🔴 严重 | `src/` 包含 6 个子包 | ✅ 已解决 |
| `utils.py` 4 个职责 | 🟡 中等 | 拆分为 4 个文件 | ✅ 已解决 |
| `pytools.py` 大杂烩 | 🟡 中等 | 数据 → `data/`, 可视化 → `viz.py` | ✅ 已解决 |
| 硬编码路径/超参 | 🟡 中等 | YAML 配置 + CLI 覆盖 | ✅ 已解决 |
| 拼写 `deicom` | 🟢 轻微 | 已修复 | ✅ 已解决 |
| 无配置系统 | 🟢 轻微 | YAML + dataclass | ✅ 已解决 |
| `transformer_model.py` 空壳 | 🟢 轻微 | 未迁移 | ✅ 已移除 |

### 保留的优点

重构过程中保留了以下设计：

- ✅ `datasets/` 已有封装 → 迁移到 `src/data/`，接口不变
- ✅ DenseUNet2d 架构实现 → 完整保留在 `models/unet.py`
- ✅ 图像域 + 投影域双分支 → 保持两个独立配置文件
- ✅ 多种训练策略的全面探索 → 8 个实验配置覆盖所有原版策略
- ✅ `docs/` 文档 → 完整迁移到 `docs/`
- ✅ 训练/推理分离 → `train.py` + `test.py`
- ✅ TensorBoard 日志 → 保留并增强
- ✅ 检查点格式 → 完全兼容

---

> **文档版本**: 2.0.0  
> **最后更新**: 2026-07-26  
> **项目地址**: `C:\imageReconstruction\AICTVer2`
