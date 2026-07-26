# AICTVer2 训练与测试指南

> AICTVer2 重构版快速上手指南。原版 AICT 有 9 个分散的训练脚本，重构后统一为 `train.py` + YAML 配置，所有实验变体通过配置切换。

---

## 0. 环境配置

### 已安装（无需操作）

| 包 | 版本 |
|---|------|
| torch | ≥2.0 |
| numpy | ≥1.24 |
| tensorboard | ≥2.15 |
| tqdm | ≥4.60 |
| pyyaml | ≥6.0 |

### 需要补装

```bash
pip install scipy scikit-image opencv-python
```

> 三个包的作用：`scipy`（图像滤波用）、`scikit-image`（测试时算 PSNR/SSIM）、`opencv-python`（高斯平滑用）。

---

## 1. 数据说明

### 1.1 图像域（PhaseCT）

数据为 `.raw` 格式的体数据，每个体数据 shape `(160, 512, 512)`，float32，小端序。

| 文件 | 用途 |
|------|------|
| `rec_pbi_fs_dec_blur_phase_100000.raw` | 模型输入（模糊相位校正后的重建） |
| `rec_no_pbi.raw` | Ground Truth 标签（无相位束硬化校正） |

读取器：`src/data/raw_reader.py`
- 按切片索引取出单张切片 `(1, 512, 512)`
- HU 归一化：`(x + 1000) / 2000`，将 `[-1000, +1000]` 映射到 `[0, 1]`，再 clip

### 1.2 投影域（ProjCT）

| 文件 | 用途 |
|------|------|
| `proj_pbi_fs_dec_blur_phase_100000.raw` | 模型输入（含伪影的投影数据） |
| `proj_no_pbi.raw` | Ground Truth 标签（无伪影的投影数据） |

- 数据形状：`(720, 128, 512)` = `(视角数, 探测器行, 探测器通道)`
- 单个样本：一个视角 `(1, 128, 512)`（二维探测器平面）
- 每病人 720 个视角，首尾均有效
- 数据范围约 `-0.066 ~ 0.276`，归一化：`clip(x, 0, 0.28) / 0.28`

读取器：`src/data/proj_reader.py`
- 使用 `np.fromfile` + seek 偏移，避免全量读取，I/O 效率高

---

## 2. 重构改动一览

### 核心变化

| 原版（AICT） | 重构版（AICTVer2） |
|---|---|
| 9 个独立训练脚本（`train_unet_2d.py`, `train_unet_proj_2d.py`, `train_unet_2d_gan.py` …） | 1 个统一入口 `train.py` + YAML 配置 |
| 2 个测试脚本（`test_raw.py`, `test_proj_raw.py`） | 1 个统一入口 `test.py --domain phase\|proj` |
| `datasets/` 目录存放读取器 | `src/data/` 子包 |
| 超参数硬编码在脚本里 | 全部在 YAML 文件 `experiments/*.yaml` |
| `utils.py` 混杂 I/O / 指标 / 增强 / 可视化 | 拆分为 `io_utils.py`, `metrics.py`, `augments.py`, `viz.py` |
| 只能改源码调参 | `--epochs`, `--batch-size`, `--lr` 等 CLI 覆盖，不用改代码 |

### 旧 -> 新 文件映射

| 旧文件 | 新文件 |
|--------|--------|
| `train_unet_2d.py` | `train.py` + `experiments/denseunet_phasect.yaml` |
| `train_unet_proj_2d.py` | `train.py` + `experiments/denseunet_projct.yaml` |
| `train_unet_2d_mixup.py` | `train.py` + `experiments/denseunet_mixup.yaml` |
| `train_unet_2d_cutmix.py` | `train.py` + `experiments/unet_cutmix.yaml` |
| `train_unet_2d_gan.py` | `train.py` + `experiments/unet_gan.yaml` |
| `train_unet_2d_vgg.py` | `train.py` + `experiments/unet_vgg.yaml` |
| `train_unet_2d_deepsup.py` | `train.py` + `experiments/unet_deepsup.yaml` |
| `train_unet_2d_sophia.py` | `train.py` + `experiments/denseunet_sophia.yaml` |
| `test_raw.py` | `test.py --domain phase` |
| `test_proj_raw.py` | `test.py --domain proj` |
| `datasets/read_raw_file.py` | `src/data/raw_reader.py` |
| `datasets/read_proj_file.py` | `src/data/proj_reader.py` |
| 每个脚本各自定义模型 | `src/models/unet.py` + `src/models/blocks.py` |

---

## 3. 项目结构

```
AICTVer2/
├── train.py                       # 训练入口（唯一）
├── test.py                        # 推理入口（唯一）
├── prepare_data.py                # 生成图像域数据索引
├── prepare_proj_data.py           # 生成投影域数据索引
├── src/
│   ├── config.py                  # YAML 配置解析
│   ├── trainer.py                 # 统一训练引擎（标准/GAN/DeepSup）
│   ├── models/                    # 网络架构
│   │   ├── unet.py                # UNet2d, DenseUNet2d, UNet2dDeepSuperviser
│   │   ├── blocks.py              # ConvBnRelu, StackEncoder/Decoder, Dense blocks
│   │   └── discriminator.py       # 判别器 + 梯度惩罚
│   ├── data/                      # 数据读取器
│   │   ├── raw_reader.py          # PhaseCT 读取器
│   │   ├── proj_reader.py         # ProjCT 读取器
│   │   └── dicom_reader.py        # DICOM 读取器（兼容旧数据）
│   ├── losses/                    # 损失函数
│   │   ├── regularizers.py        # CharbonnierLoss, TVLoss
│   │   ├── ssim.py                # SSIM 损失/指标
│   │   └── perceptual.py          # VGG 感知损失
│   ├── utils/
│   │   ├── io_utils.py            # 模型保存/加载
│   │   ├── metrics.py             # AverageMeter, normalization, PSNR
│   │   ├── augments.py            # MixUp, CutMix
│   │   └── viz.py                 # 高斯平滑, patch提取
│   └── optimizers/
│       └── sophia.py              # SophiaG 优化器
├── experiments/                   # 8 个预置 YAML 配置
│   ├── denseunet_phasect.yaml
│   ├── denseunet_projct.yaml
│   ├── denseunet_mixup.yaml
│   ├── unet_cutmix.yaml
│   ├── unet_gan.yaml
│   ├── unet_vgg.yaml
│   ├── unet_deepsup.yaml
│   └── denseunet_sophia.yaml
└── runs/                          # 训练输出（checkpoints + 日志）
```

---

## 4. 运行步骤

### 4.1 生成数据索引（只需执行一次）

**图像域（PhaseCT）：**
```bash
python prepare_data.py --data-dir /path/to/UIH_data --output ./
```
生成：
- `train_img.txt` — 11 个病人 × 158 张切片 = 1738 条
- `valid_img.txt` — 2 个病人 × 158 张切片 = 316 条

> 首尾切片（索引 0 和 159）已默认跳过。可通过 `--num-slices` 和 `--slice-start` 调整。

**投影域（ProjCT）：**
```bash
python prepare_proj_data.py --data-dir /path/to/UIH_data --output ./
```
生成：
- `train_proj_img.txt` — 11 个病人 × 720 视角 = 7920 条
- `valid_proj_img.txt` — 2 个病人 × 720 视角 = 1440 条

### 4.2 训练

```bash
# PhaseCT（图像域）—— DenseUNet2d 基线
python train.py --config experiments/denseunet_phasect.yaml

# ProjCT（投影域）
python train.py --config experiments/denseunet_projct.yaml

# MixUp 增强
python train.py --config experiments/denseunet_mixup.yaml

# CutMix 增强
python train.py --config experiments/unet_cutmix.yaml

# GAN（WGAN-GP）对抗训练
python train.py --config experiments/unet_gan.yaml

# VGG 感知损失
python train.py --config experiments/unet_vgg.yaml

# Deep Supervision（多尺度监督）
python train.py --config experiments/unet_deepsup.yaml

# SophiaG 优化器
python train.py --config experiments/denseunet_sophia.yaml
```

#### CLI 参数覆盖（不用改 YAML）

```bash
# 修改 epoch 数
python train.py --config experiments/denseunet_phasect.yaml --epochs 300

# 修改 batch size（显存不够时）
python train.py --config experiments/denseunet_phasect.yaml --batch-size 4

# 修改学习率
python train.py --config experiments/denseunet_phasect.yaml --lr 0.0005

# 指定 GPU
python train.py --config experiments/denseunet_phasect.yaml --gpu 1

# 修改保存/日志目录
python train.py --config experiments/denseunet_phasect.yaml --save-dir /tmp/checkpoints --log-dir /tmp/logs
```

#### 断点续训

```bash
# 自动加载 runs/ 下最新的 .dat checkpoint 继续训练
python train.py --config experiments/denseunet_phasect.yaml --resume
```

训练输出位置（由 YAML 中 `training.save_dir` 和 `training.log_dir` 控制）：

| 内容 | 默认路径 |
|------|---------|
| Checkpoint 文件 | `./runs/<实验名>/checkpoints/model_at_epoch_XXX.dat` |
| TensorBoard 日志 | `./runs/<实验名>/logs/<时间戳>/` |

### 4.3 监控

```bash
tensorboard --logdir ./runs
```

浏览器打开 TensorBoard，可查看：
- Loss 曲线（训练 MSE + 验证 MSE/PSNR/SSIM）
- 输入图像（ldct）、标签（reference）、预测结果（predict）、残差图（residual）

### 4.4 测试（推理）

```bash
# PhaseCT 推理
python test.py --config experiments/denseunet_phasect.yaml \
    --checkpoint ./runs/DenseUNet_PhaseCT/checkpoints/model_at_epoch_200.dat \
    --data-dir /path/to/data \
    --output ./results/

# ProjCT 推理
python test.py --config experiments/denseunet_projct.yaml \
    --checkpoint ./runs/DenseUNet_ProjCT/checkpoints/model_at_epoch_50.dat \
    --data-dir /path/to/data \
    --output ./results_proj/ \
    --domain proj
```

`test.py` 会自动：
1. 根据 `--domain` 选择正确的读取方式（PhaseCT 切片循环 / ProjCT 视角循环）
2. 逐切片/视角推理
3. 保存预测体数据到输出目录
4. 计算 PSNR 和 SSIM 并打印

可选参数：
```bash
# 指定特定病人（可多个）
python test.py ... --patient "patient_A" "patient_B"

# 指定 GPU
python test.py ... --gpu 0
```

---

## 5. 配置详解

### 5.1 YAML 配置结构

以 `denseunet_phasect.yaml` 为例：

```yaml
experiment:
  name: DenseUNet_PhaseCT    # 实验名称（决定日志/checkpoint 子目录）
  seed: 1234                  # 随机种子

data:
  reader: raw                 # 数据读取器：raw | dicom | proj
  train_txt: ./train_img.txt  # 训练集索引文件
  valid_txt: ./valid_img.txt  # 验证集索引文件
  batch_size: 12              # 训练 batch size
  valid_batch_size: 1         # 验证 batch size（通常为 1）
  num_workers: 8              # 数据加载线程（Windows 设为 0）
  shuffle: true               # 是否打乱

model:
  name: DenseUNet2d           # 模型：DenseUNet2d | UNet2d | UNet2dDeepSuperviser
  in_chl: 1                   # 输入通道数
  out_chl: 1                  # 输出通道数
  model_chl: 32               # 基础特征通道数

loss:
  mse_weight: 1.0             # MSE 损失权重
  charbonnier_weight: 0.0     # Charbonnier 损失权重
  ssim_weight: 0.0            # SSIM 损失权重（1 - SSIM）
  vgg_weight: 0.0             # VGG 感知损失权重
  tv_weight: 0.0              # 总变分正则化权重
  adv_weight: 0.0             # 对抗损失权重（>0 自动启用 GAN）
  deep_supervision: false      # 多尺度深度监督

optimizer:
  name: Adam                  # Adam | AdamW | SophiaG
  lr: 0.0001
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0

scheduler:
  name: StepLR                # StepLR | MultiStepLR | CosineAnnealingLR
  step_size: 2
  gamma: 0.95

gan:                          # 仅 adv_weight > 0 时生效
  d_lr: 0.0001                # 判别器学习率
  d_betas: [0.5, 0.9]        # 判别器 Adam beta
  gp_weight: 10.0             # 梯度惩罚系数

training:
  epochs: 200
  save_dir: ./runs/DenseUNet_PhaseCT
  log_dir: ./runs/DenseUNet_PhaseCT
  log_interval: 20            # 日志打印间隔（步数）
  viz_interval: 1             # 图像日志间隔（epochs）
  save_interval: 1            # 保存 checkpoint 间隔（epochs）
```

### 5.2 损失函数组合规则

各损失函数通过权重激活，互不冲突，可任意组合：

| 损失函数 | 权重字段 | 适用场景 |
|---------|---------|---------|
| MSE | `mse_weight` | 基线训练，像素级 L2 距离 |
| Charbonnier | `charbonnier_weight` | 对 outliers 更鲁棒（近似 L1） |
| SSIM | `ssim_weight` | 结构相似性优化（1 - SSIM） |
| VGG | `vgg_weight` | 感知质量优化，高频细节保留 |
| TV | `tv_weight` | 平滑正则化，抑制噪声 |
| GAN (Adv) | `adv_weight` | 对抗训练，生成更真实的纹理 |

> 例如 `unet_vgg.yaml`：`mse_weight: 1.0` + `vgg_weight: 0.1`，兼顾 MSE 和感知质量。

---

## 6. 实验变体速查

| 配置 | 模型 | 优化器 | 特殊损失 | 增强 | 数据域 |
|------|------|--------|---------|------|--------|
| `denseunet_phasect` | DenseUNet2d | Adam | MSE | — | PhaseCT |
| `denseunet_projct` | DenseUNet2d | Adam | MSE | — | ProjCT |
| `denseunet_mixup` | DenseUNet2d | AdamW | MSE | MixUp(α=0.2) | PhaseCT |
| `unet_cutmix` | UNet2d | Adam | MSE | CutMix V2 | PhaseCT |
| `unet_gan` | UNet2d | Adam | MSE + Adv(0.01) | — | PhaseCT |
| `unet_vgg` | UNet2d | Adam | MSE + VGG(0.1) | — | PhaseCT |
| `unet_deepsup` | UNet2dDeepSuperviser | Adam | MSE(多尺度) | MixUp | PhaseCT |
| `denseunet_sophia` | DenseUNet2d | SophiaG | MSE | — | PhaseCT |

---

## 7. 快速参考

| 操作 | 命令 |
|------|------|
| 生成 PhaseCT 索引 | `python prepare_data.py --data-dir <path> --output ./` |
| 生成 ProjCT 索引 | `python prepare_proj_data.py --data-dir <path> --output ./` |
| PhaseCT 训练 | `python train.py -c experiments/denseunet_phasect.yaml` |
| ProjCT 训练 | `python train.py -c experiments/denseunet_projct.yaml` |
| GAN 训练 | `python train.py -c experiments/unet_gan.yaml` |
| 断点续训 | `python train.py -c <config> --resume` |
| CLI 覆盖参数 | `python train.py -c <config> --epochs 300 --lr 0.0005` |
| 监控 | `tensorboard --logdir ./runs` |
| PhaseCT 推理 | `python test.py -c <config> --checkpoint <ckpt> --data-dir <path>` |
| ProjCT 推理 | `python test.py -c <config> --checkpoint <ckpt> --data-dir <path> --domain proj` |

---

## 8. 常见调整

- **减慢过拟合**：在 YAML 中设 `loss.tv_weight: 0.001` 或 `augment.name: mixup`
- **显存不够**：`python train.py -c <config> --batch-size 4`；Windows 还需要把 YAML 中的 `num_workers` 改为 0
- **收敛太慢**：调大 `optimizer.lr`（如 0.0002）或换用 CosineAnnealingLR + 更多 epochs
- **GAN 训练不稳定**：增大 `gan.gp_weight`（梯度惩罚系数）或调小 `loss.adv_weight`
- **仅跑验证/调试**：临时改 YAML 中 `training.epochs: 1`，或直接在命令行 `--epochs 1`
- **多 GPU 训练**：`CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c <config>`（自动 DataParallel）
- **数据量太大？** `prepare_data.py` 中 `--num-slices` 减半（如 79），或 `--slice-start` 改为 0 并 `--num-slices 80`（取前 80 张）
- **Windows 用户注意**：所有 YAML 中 `num_workers` 必须设为 0

---

## 9. 从原版 AICT 迁移

如果之前用原版 AICT 训练过，以下注意：

1. **Checkpoint 完全兼容**：模型架构相同，`.dat` 文件可直接加载
   ```python
   from src.models import DenseUNet2d
   from src.utils.io_utils import load_model
   model = DenseUNet2d()
   load_model(model, torch.load("old_checkpoint.dat"))
   ```

2. **数据索引文件格式不变**：仍为 `每行：输入路径 标签路径 索引`，旧文件可直接用

3. **输出目录结构变化**：原版 checkpoints/logs 在同一目录混杂，新版结构化到 `runs/<实验名>/checkpoints/` 和 `runs/<实验名>/logs/<时间戳>/`

4. **配置迁移**：将原版脚本中硬编码的参数（batch_size, lr, epochs 等）填入对应 YAML 字段即可，参考 `experiments/` 下的示例
