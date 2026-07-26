# 项目架构分析与重构建议

> 分析日期: 2026-07-26
> 项目: AICT (AI CT Image Reconstruction)

---

## 1. 项目概述

AICT 是一个基于深度学习的 CT 图像重建项目，专注于使用神经网络对低剂量 CT (LDCT) 图像进行降噪/增强。项目包含**两大分支**：

| 分支 | 领域 | 输入 → 输出 | 数据形状 |
|------|------|-------------|---------|
| **图像域 (PhaseCT)** | 重建后的 CT 切片 | `rec_pbi_*.raw` → `rec_no_pbi.raw` | (160, 512, 512) 体数据 |
| **投影域 (ProjCT)** | 原始投影数据 | `proj_pbi_*.raw` → `proj_no_pbi.raw` | (720, 128, 512) 视角序列 |

核心是 **DenseUNet2d** 架构，配合多种训练策略（数据增强、GAN、感知损失、深度监督、不同优化器）。

---

## 2. 当前目录结构

```
AICT/
├── datasets/                      # ✅ 已有包结构
│   ├── __init__.py                # 导出三个 reader
│   ├── read_raw_file.py           # raw_reader: 相片CT切片读取
│   ├── read_dicom_file.py         # dicom_reader: DICOM文件读取
│   └── read_proj_file.py          # proj_reader: 投影域视角读取
│
├── docs/                          # ✅ 文档目录
│   ├── DATASET_FORMAT.md
│   ├── MSE_RMSE.md
│   ├── PROJECTION_DOMAIN_CHANGES.md
│   ├── TRAINING_GUIDE.md
│   └── ARCHITECTURE_ANALYSIS.md   # ← 本文档
│
├── results/                       # 图像域推理结果 (.raw)
├── results_proj/                  # 投影域推理结果 (.raw)
├── runs/                          # 训练输出
│   ├── DenseUNet_PhaseCT/         #   图像域检查点 + TensorBoard
│   └── DenseUNet_ProjCT/          #   投影域检查点 + TensorBoard
│
├── models.py                      # (核心) 模型架构
├── losses.py                      # (核心) 损失函数
├── utils.py                       # (核心) 工具函数
├── pytools.py                     # (核心) 数据I/O与可视化
├── sophia.py                      # (核心) SophiaG优化器
├── transformer_model.py           # ⚠️ 空壳, 未实现
│
├── train_unet_2d.py               # 图像域训练 (PhaseCT)
├── train_unet_proj_2d.py          # 投影域训练 (ProjCT)
├── train_unet_2d_mixup.py         # + MixUp
├── train_unet_2d_cutmix.py        # + CutMix
├── train_unet_2d_gan.py           # + GAN (生成对抗)
├── train_unet_2d_vgg.py           # + VGG感知损失
├── train_unet_2d_vgg_mgpu.py      # + VGG多GPU
├── train_unet_2d_mixup_deepsurperviser.py  # + 深度监督 + MixUp
├── train_unet_2d_mixup_sophia.py  # + Sophia优化器 + MixUp
│
├── test.py                        # DICOM测试/推理
├── test_raw.py                    # 相片CT测试/推理
├── test_proj_raw.py               # 投影域测试/推理
│
├── prepare_data.py                # 相片CT数据列表生成
├── prepare_proj_data.py           # 投影域数据列表生成
├── mk_data_list.py                # AAPM-DICOM数据列表生成
│
├── check_proj_data.py             # 投影数据shape探测
├── analyze_proj_range.py          # 投影数据范围分析
│
├── train_img.txt                   # 训练数据索引 (绝对路径)
├── valid_img.txt                   # 验证数据索引 (绝对路径)
├── requirements.txt                # 依赖
├── .gitignore
└── ...其他配置文件
```

---

## 3. 数据流架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                                │
│                                                                  │
│  prepare_data.py / prepare_proj_data.py                          │
│    ↓                                                             │
│  train_img.txt / valid_img.txt                                   │
│  (每行: input_path label_path slice_index)                       │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DataLoader 层                              │
│                                                                  │
│  raw_reader / dicom_reader / proj_reader                         │
│    - 按索引读取单个切片/视角                                      │
│    - 归一化到 [0, 1]                                             │
│    - 返回 {"ldct": tensor, "rdct": tensor}                       │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                       训练循环 (train/valid)                      │
│                                                                  │
│  1. 可选: MixUp/CutMix 数据增强                                  │
│  2. 模型前向: model(ldct) → predict                              │
│  3. 损失计算: MSE + Charbonnier + SSIM + VGG + GAN              │
│  4. 反向传播 + 优化器步进                                        │
│  5. TensorBoard 日志记录                                         │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                       模型架构 (models.py)                        │
│                                                                  │
│  ┌─ ConvBnRelu2d (基础卷积块)                                    │
│  ├─ StackEncoder / StackDecoder (标准U-Net编解码块)              │
│  ├─ StackDenseEncoder2d / StackResDecoder2d (DenseU-Net变体)     │
│  ├─ UNet2d — 标准U-Net (残差连接输出)                            │
│  ├─ UNet2dDeepSuperviser — 含中间监督输出的U-Net                 │
│  ├─ DenseUNet2d — DenseNet风格U-Net (推荐)                      │
│  └─ Discriminator2D — GAN判别器                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心依赖关系

```
train_unet_2d.py ───────────────────────┐
train_unet_proj_2d.py ──────────────────┤
train_unet_2d_mixup.py ─────────────────┤
... (所有训练脚本) ─────────────────────┤
                                         │
test_raw.py ─────────────────────────────┤
test_proj_raw.py ────────────────────────┤
                                         ▼
                              ┌─────────────────────┐
                              │   datasets/          │
                              │   (数据读取)         │
                              └─────────┬───────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────┐
         ▼                              ▼                      ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  models.py   │  │  losses.py   │  │  utils.py    │  │  pytools.py  │
  │  (网络结构)   │  │  (损失函数)   │  │ (工具函数)    │  │ (I/O/可视化) │
  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

---

## 5. 主要问题

### 5.1 大量代码重复 — 训练脚本 🔴 严重

**9个训练脚本**中，`train()` 和 `valid()` 函数高度重复（每次都是几乎完全一样的循环），区别仅在于：

| 脚本文件名 | 模型 | 数据增强 | 损失函数 | 优化器 | 多GPU |
|-----------|------|---------|---------|--------|-------|
| `train_unet_2d` | DenseUNet2d | ❌ | MSE+Charbonnier | Adam | DataParallel |
| `train_unet_proj_2d` | DenseUNet2d | ❌ | MSE+Charbonnier | Adam | DataParallel |
| `train_unet_2d_mixup` | DenseUNet2d | MixUp | MSE+Charbonnier | AdamW | DataParallel |
| `train_unet_2d_cutmix` | UNet2d | CutMixV2 | MSE+Charbonnier | AdamW | DataParallel |
| `train_unet_2d_gan` | UNet2d+Disc | ❌ | MSE+WGAN-GP | Adam | ❌ |
| `train_unet_2d_vgg` | UNet2d | ❌ | MSE+VGG+SSIM | Adam | ❌ |
| `train_unet_2d_vgg_mgpu` | UNet2d | ❌ | MSE+VGG+SSIM | Adam | 手动多卡 |
| `train_unet_2d_mixup_deepsurperviser` | UNet2dDeepSuperviser | MixUp | 多尺度MSE | AdamW | DataParallel |
| `train_unet_2d_mixup_sophia` | DenseUNet2d | MixUp | MSE+Charbonnier | SophiaG | DataParallel |

**问题所在**: 训练循环逻辑重复了9遍，每次只是超参、增强、损失的不同组合。后续要改（如添加验证指标、调整日志格式）需要改9个文件。

### 5.2 核心库散落在根目录 🔴 严重

`models.py`, `losses.py`, `utils.py`, `pytools.py`, `sophia.py` 都在项目根目录，没有放在 `src/` 或独立包中。导致：
- `from models import *` 这种通配符导入（不可控、易冲突）
- 大型项目目录混乱，根目录下有 20+ 文件
- `transformer_model.py` 是一个只有 import 的空文件（未完成实现）

### 5.3 `utils.py` 职责过重 🟡 中等

`utils.py` 包含以下不相关的功能：
- 模型保存/加载 (save_model / load_model variants)
- SSIM 计算指标 (SSIM类 + ssim函数)
- 数据增强 (MixUp_AUG, CutMix_AUG, CutMix_AUG_V2)
- 高斯滤波 (gaussian_smooth)
- Patches 提取 (extract_patches_online)
- 计算工具 (AverageMeter, normalization)

### 5.4 `pytools.py` 大杂烩 🟡 中等

`pytools.py` 包含：
- raw数据读取
- DICOM读取
- 泊松噪声模拟
- 3D可视化 (plotImg3D 类)
- 目录工具

### 5.5 硬编码路径 🟡 中等

- `train_img.txt` / `valid_img.txt` 包含 AutoDL 服务器绝对路径
- 每个训练脚本底部 hardcode 了 `result_path`, `save_dir`, `batch_size`, `lr`
- 测试脚本中 hardcode 了 `model_dir`, `data_root`

### 5.6 命名不一致 🟢 轻微

- `read_dicom_file.py` 但在 `train_unet_2d_mixup.py` 中写作 `read_deicom_file`（拼写错误）
- 数据 key 使用 `ldct`/`rdct`，但实际是 phase 数据而非 CT 值

### 5.7 缺少统一配置系统 🟢 轻微

没有使用 yaml/json 配置或 hydra/argparse 标准，导致实验管理困难（超参数散落在各脚本中）。

### 5.8 数据索引文件是绝对路径 🟢 轻微

`train_img.txt` 使用 AutoDL 服务器路径，无法在本地 Windows 直接使用。

---

## 6. 重构建议

### 6.1 🔴 优先级最高 — 提取训练引擎为可复用模块

**目标**: 消除9个训练脚本中90%的重复代码

**方案**: 将核心训练循环提取到 `src/trainer.py`，用配置对象控制行为差异：

```
src/
├── trainer.py          # 通用训练/验证循环
├── config.py           # 配置数据类 (dataclass)
└── __init__.py
```

各个实验只需写一个简短的配置文件：

```python
# experiments/denseunet_phasect.py
from src.config import Config, DataConfig, OptimConfig

config = Config(
    name="DenseUNet_PhaseCT",
    data=DataConfig(reader="raw", train_txt="./train_img.txt"),
    model="DenseUNet2d",
    optimizer=OptimConfig(name="adam", lr=1e-4),
    # ...
)
```

而不是每个实验复制整个 `train()` + `valid()` + `if __name__ == "__main__"` 三件套。

### 6.2 🔴 优先级高 — 重构包结构

```
AICT/
├── src/                          # 核心源代码
│   ├── __init__.py
│   ├── models/                   # 模型架构
│   │   ├── __init__.py
│   │   ├── unet.py               # UNet2d, DenseUNet2d
│   │   ├── discriminator.py      # Discriminator2D
│   │   └── blocks.py             # ConvBnRelu2d, StackEncoder, etc.
│   ├── losses/                   # 损失函数
│   │   ├── __init__.py
│   │   ├── perceptual.py         # VGG loss
│   │   ├── regularizers.py       # TV loss, CharbonnierLoss
│   │   └── ssim.py               # SSIM
│   ├── data/                     # 数据加载 (从 datasets/ 迁移)
│   │   ├── __init__.py
│   │   ├── raw_reader.py
│   │   ├── dicom_reader.py
│   │   └── proj_reader.py
│   ├── utils/                    # 工具函数拆分
│   │   ├── __init__.py
│   │   ├── io_utils.py           # 模型保存/加载
│   │   ├── metrics.py            # SSIM, PSNR, AverageMeter
│   │   ├── augments.py           # MixUp, CutMix
│   │   └── viz.py                # gaussian_smooth, patches
│   ├── optimizers/               # 优化器
│   │   ├── __init__.py
│   │   └── sophia.py
│   └── trainer.py                # 统一训练引擎
│
├── experiments/                  # 实验配置
│   ├── denseunet_phasect.yaml    # 或 .py
│   ├── denseunet_projct.yaml
│   ├── unet_mixup.yaml
│   └── unet_gan.yaml
│
├── datasets/                     # 保留或迁移到 src/data/
├── docs/
├── results/
├── results_proj/
├── runs/
│
├── train.py                      # 入口: python train.py --config experiments/xxx.yaml
├── test.py                       # 入口: python test.py --config ...
├── prepare_data.py               # 数据准备 (可保留根目录)
│
├── pyproject.toml                # 项目元数据 (替代 requirements.txt)
└── README.md
```

### 6.3 🟡 中等优先级 — 拆分 `utils.py` 和 `pytools.py`

- **SSIM** → `src/utils/metrics.py`（与损失函数中的 SSIM 统一）
- **MixUp/CutMix** → `src/utils/augments.py`
- **Model I/O** → `src/utils/io_utils.py`
- **AverageMeter** → `src/utils/metrics.py`
- **可视化** → `src/utils/viz.py` 或独立脚本
- **数据I/O** → 已有 `datasets/`，无需迁移

### 6.4 🟡 中等优先级 — 引入配置系统

推荐使用 **YAML 配置**（轻量 + 可读）：

```yaml
# experiments/denseunet_phasect.yaml
experiment:
  name: DenseUNet_PhaseCT
  seed: 1234

data:
  reader: raw
  train_txt: ./train_img.txt
  valid_txt: ./valid_img.txt
  batch_size: 12
  num_workers: 8

model:
  name: DenseUNet2d
  in_chl: 1
  out_chl: 1
  model_chl: 32

loss:
  mse_weight: 1.0
  charbonnier_weight: 0.0
  ssim_weight: 0.0

optimizer:
  name: Adam
  lr: 0.0001

scheduler:
  name: StepLR
  step_size: 2
  gamma: 0.95

training:
  epochs: 200
  save_dir: ./runs/DenseUNet_PhaseCT/checkpoints/
  log_dir: ./runs/DenseUNet_PhaseCT/logs/
```

### 6.5 🟢 低优先级 — 路径与配置解耦

- 数据索引文件使用**相对路径**或**占位符**，运行时通过命令行参数替换
- 在各实验脚本中统一使用 `argparse` 或 hydra 注入数据根目录

---

## 7. 重构路线图

```
Phase 1 (1-2天) — 提取训练引擎
├── 创建 src/trainer.py
├── 统一 train() + valid() 循环
├── 引入 Config dataclass
└── 验证: 重构后训练结果与原版一致

Phase 2 (1-2天) — 模块拆分
├── 拆分 utils.py → src/utils/*
├── 拆分 pytools.py → 各模块
├── 清理 models.py → src/models/
└── 验证: 所有 import 路径正确

Phase 3 (半天) — 配置系统
├── 引入 YAML 配置
├── 重写 train.py 入口
└── 迁移所有实验到配置文件

Phase 4 (可选) — 清理数据路径
├── 数据索引路径参数化
├── 清理未完成的 transformer_model.py
├── 修复拼写错误 "deicom" → "dicom"
└── 添加 setup.py / pyproject.toml
```

---

## 8. 技术债务总结

| 类别 | 严重程度 | 影响范围 |
|------|---------|---------|
| 训练循环重复 ×9 | 🔴 严重 | 维护困难, 改一个特性需改9处 |
| 根目录文件杂乱 | 🔴 严重 | import 混乱, 可读性差 |
| utils.py 职责过载 | 🟡 中等 | 模块间隐式耦合 |
| pytools.py 大杂烩 | 🟡 中等 | 数据I/O与可视化耦合 |
| 硬编码路径/超参 | 🟡 中等 | 不可移植, 实验难复现 |
| 命名不一致 | 🟢 轻微 | 可读性稍差 |
| 缺失配置系统 | 🟢 轻微 | 实验管理不便 |
| 空壳文件 transformer_model.py | 🟢 轻微 | 误导 |

---

## 9. 当前项目优势（值得保留）

- ✅ `datasets/` 已有良好封装（`__init__.py` 导出三个 reader）
- ✅ DenseUNet2d 架构实现清晰
- ✅ 图像域 + 投影域双分支设计合理
- ✅ 多种数据增强/损失/优化器组合的探索很全面
- ✅ `docs/` 目录已有基础文档
- ✅ 训练/推理分离，结果按域组织（`results/` + `results_proj/`）
- ✅ TensorBoard 日志规范

---

> **笔者建议**: 优先做 Phase 1（训练引擎复用）和 Phase 2（模块拆分），这两项能立即改善开发效率。
> Phase 1 完成后，新增实验只需写几行配置，而不是复制整个训练脚本。
