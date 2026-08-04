# 投影域训练改造方案

> 从**图像域**（重建图像）切换到**投影域**（投影数据）训练的完整改动分析。

---

## 一、核心差异概览

| 项目 | 图像域（当前） | 投影域（目标） |
|------|---------------|---------------|
| 输入文件 | `rec_pbi_fs_dec_blur_phase_100000.raw` | `proj_pbi_fs_dec_blur_phase_100000.raw` |
| 目标文件 | `rec_no_pbi.raw` | `proj_no_pbi.raw` |
| 数据形状 | `(160, 512, 512)` → 切片 `(512, 512)` | `(720, 128, 512)` → 视角 `(128, 512)` |
| 每病人样本数 | 158 张切片 | 720 个视角 |
| 数据范围 | HU 值（-1000~1000，物理有界） | 原始衰减值（非负浮点，无固定上界） |
| 归一化方式 | `(x + 1000) / 2000` 固定映射到 [0,1] | 需重新设计（见下文） |
| 模型输入尺寸 | `(1, 512, 512)` | `(1, 128, 512)` |
| 评估指标含义 | PSNR/SSIM/MSE 基于 HU，物理可解释 | 基于投影衰减值，需谨慎解读 |

---

## 二、数据形状与切片策略

投影数据 `(720, 128, 512)` 的三个维度：

```
(720, 128, 512)
  │    │    └── 探测器通道 (detector channel) — 512 通道
  │    └── 探测器行 (detector row) — 128 行
  └── 视角 (view / angle) — 720 个等角度间隔
```

### 推荐的 2D 切片方式

**按视角切**（每个视角 = 一张 `128×512` 的二维探测器平面）：

```
输入: proj_input[v]  → 形状 (128, 512) — 第 v 个视角
目标: proj_target[v] → 形状 (128, 512) — 第 v 个视角
```

- 每病人 720 个样本（vs 图像域每病人 158 张切片）
- 模型输入通道数 `in_chl=1` 无需改动
- DenseUNet2d 经过 4 次 stride=2 下采样后：`128→8`, `512→32`，尺寸完全兼容

### 替代方案（不推荐）

**按探测器行切**（每个 sinogram = 一张 `512×720` 的图像）：
- 每病人仅 128 个样本，且 `720` 维度不是 16 的倍数（`720/16=45`），需 padding 或调整网络结构
- 不推荐，改动量大

---

## 三、归一化方案（已确认数据范围）

已确认投影数据范围：**-0.066 ~ 0.276**（全数据集 13 个病人的统计结果）。

这意味着数据很可能已经是 $I/I_0$（空气校正后的透射比值）或已经做过预处理。数值本身接近 [0, 1]，无需复杂变换。

### 推荐方案：直接 clip + 小幅缩放

```python
# 数据范围约 [-0.066, 0.276]，只需：
# 1. 截断负值（物理上非负）
# 2. 缩放到 [0, 1]（用 0.28 作为安全分母）
proj_norm = np.clip(proj, 0, 0.28) / 0.28
```

或者更保守的 min-max：

```python
# 每个体数据独立归一化
p_min, p_max = proj.min(), proj.max()
proj_norm = (proj - p_min) / (p_max - p_min + 1e-8)
```

**推荐用 min-max**，简单、适应性强、不丢失动态范围。

---

## 四、需要改动的文件

### 1. `prepare_data.py` — 新增投影域版本

建议新建 `prepare_proj_data.py`，改动点：

| 改动 | 说明 |
|------|------|
| 路径后缀 | `rec_pbi_...` → `proj_pbi_...` |
| 目标后缀 | `rec_no_pbi.raw` → `proj_no_pbi.raw` |
| 切片索引 | 不用 slice_idx，改用 view_idx（0~719） |
| 输出文件 | 生成 `train_proj_img.txt` / `valid_proj_img.txt` |
| 每行格式 | `输入路径 目标路径 视角索引` |

```python
# 每行格式
# /path/to/proj_pbi_...  /path/to/proj_no_pbi.raw  0
# ...
# /path/to/proj_pbi_...  /path/to/proj_no_pbi.raw  719
```

### 2. `datasets/read_raw_file.py` — 新增投影域 DataLoader

建议新建 `datasets/read_proj_file.py`，或给现有 `raw_reader` 加参数：

```python
class proj_reader(Dataset):
    def __init__(self, paired_img_txt, proj_min=0.0, proj_max=None):
        # 读取索引文件，每行：input_path label_path view_idx
        ...
    
    def __getitem__(self, index):
        # 1. 读取投影体数据 reshape(720, 128, 512)
        # 2. 按 view_idx 取第 v 个视角: proj[v] → (128, 512)
        # 3. 归一化到 [0, 1]
        # 4. 添加通道维度: (1, 128, 512)
        # 5. 返回 {"ldct": input_tensor, "rdct": label_tensor}
        #    （键名保持 ldct/rdct 以兼容 train_unet_2d.py）
```

**关键注意**：重建图像是 `(160, 512, 512)` → reshape 后取 `[slice_idx]`；投影数据是 `(720, 128, 512)` → reshape 后取 `[view_idx]`，切片维度不同。

### 3. `train_unet_2d.py` — 改动量最小

| 位置 | 改动 |
|------|------|
| import | 从 `read_proj_file` 导入 `proj_reader` 替代 `raw_reader` |
| 数据集加载 | `raw_reader('./train_img.txt')` → `proj_reader('./train_proj_img.txt')` |
| 同上 valid | 同理 |
| batch_size | 建议调大（`(1, 128, 512)` 比 `(1, 512, 512)` 小 4 倍，显存占用更低） |
| 输出路径 | 建议换目录保存，避免与图像域 checkpoint 混淆 |
| TensorBoard 图像 | 可沿用，投影图像可视化不如重建图像直观，但仍有监控价值 |

**不需要改动的部分**：
- 模型定义（DenseUNet2d，in_chl=1, out_chl=1）
- Loss 函数（MSELoss + CharbonnierLoss）
- Optimizer / Scheduler
- 训练 / 验证循环逻辑

### 4. `test_raw.py` — 新增投影域测试脚本

建议新建 `test_proj_raw.py`，主要改动：

| 改动 | 说明 |
|------|------|
| 数据读取 | 读 `proj_*.raw` 文件，reshape 为 `(720, 128, 512)` |
| 逐视角推理 | 循环 `range(720)`，每步取 `proj[v]` |
| 归一化/反归一化 | 使用投影域的归一化参数（非 HU 范围） |
| 保存结果 | 保存 `*_input.raw`, `*_gt.raw`, `*_pred.raw` (720×128×512) |
| 评估指标 | 计算 PSNR/SSIM/MSE/RMSE/MAE，但 **不解释为 HU** |

### 5. `TRAINING_GUIDE.md` — 更新文档

补充投影域训练的完整步骤说明。

---

## 五、DenseUNet2d 输入尺寸兼容性验证

```
输入 (1, 128, 512)
  ↓ begin (Conv 3×3, stride 1)
→ (32, 128, 512)
  ↓ down1 (pool /2)
→ (32, 64, 256)
  ↓ down2 (pool /2)
→ (64, 32, 128)
  ↓ down3 (pool /2)
→ (128, 16, 64)
  ↓ down4 (pool /2)
→ (256, 8, 32)
  ↓ center (dense block)
→ (512, 8, 32)
  ↓ up4 (upsample ×2)
→ (256, 16, 64)
  ↓ ...
  ↓ up1 (upsample ×2)
→ (32, 128, 512)
  ↓ end (conv 1×1)
→ (1, 128, 512)
```

✅ 所有分辨率都是整数，`ceil_mode=True` 确保边界对齐。

---

## 六、数据量对比与训练效率

| 指标 | 图像域 | 投影域 |
|------|--------|--------|
| 每病人样本数 | 158 | 720 |
| 13 病人总样本 | 11 训练 × 158 = 1738 | 11 训练 × 720 = 7920 |
| 单样本尺寸 | `(1, 512, 512)` | `(1, 128, 512)` |
| 每 epoch 前向计算量 | 基准 | 约 0.25×（面积小 4 倍） |
| batch_size 建议 | 12 | 24~48 |
| 训练时间估计 | 基准 | 相近或略快（样本多但每样本小） |

### Epoch 数建议

由于投影域每 epoch 步数更多（7920 vs 1738），且每个样本包含的信息量不同，建议：

- 先跑 50~100 epoch 观察 loss 收敛曲线
- 如果收敛慢，适当增加 epoch 到 200~300

---

## 七、需要注意的问题

### 1. 投影数据的物理含义 ✅ 已确认

实测范围 `-0.066 ~ 0.276`（全数据集统计），数值极小，说明数据已经是 **$I/I_0$ 比值**或已预处理归一化，可以直接 clip 到 [0,1] 范围使用。

### 2. 首尾视角边界效应 ✅ 已确认

不需剔除，使用全部 720 个视角 `range(0, 720)`。

### 3. 与图像域 checkpoint 的管理

投影域训练需全新的 checkpoint（不能复用图像域预训练权重，因为输入分布完全不同）。建议：

- 训练结果保存到 `./runs/DenseUNet_ProjCT/` 目录
- 或者 `./runs/DenseUNet_PhaseCT_proj/`
- 避免与图像域文件混淆

### 4. 评估指标的可解释性

投影域的 MSE/RMSE/PSNR 是在**投影衰减值**上计算的，不是 HU，所以：

- **不要**将投影域 RMSE 解释为 "多少 HU"
- 投影域指标用于监控训练收敛和相对比较（不同模型在投影域的表现）
- 最终图像质量仍需通过重建算法将投影转回图像域后用 HU 评估

---

## 八、改动步骤总结（建议执行顺序）

```
步骤 1 — 数据分析
  ├ 读一个 proj_no_pbi.raw 样本，确认数据范围
  ├ 确定归一化策略（全局 min-max / 每体独立 / log 变换）
  └ 确认首尾视角是否需要剔除

步骤 2 — 准备数据索引
  ├ 新建 prepare_proj_data.py（基于 prepare_data.py 修改）
  └ 运行生成 train_proj_img.txt / valid_proj_img.txt

步骤 3 — 实现投影域 DataLoader
  ├ 新建 datasets/read_proj_file.py
  └ 实现 proj_reader 类

步骤 4 — 修改训练脚本
  ├ 复制 train_unet_2d.py → train_unet_proj_2d.py
  ├ 修改 import 和数据集加载部分
  └ 修改保存目录（避免覆盖图像域 checkpoint）

步骤 5 — 修改测试脚本
  ├ 复制 test_raw.py → test_proj_raw.py
  └ 适配投影数据形状、归一化和评估

步骤 6 — 训练
  ├ 先跑 10 epoch 确认 loss 正常下降
  └ 再跑完整训练

步骤 7 — 更新文档
  └ 更新 TRAINING_GUIDE.md 加入投影域训练说明
```

---

## 九、训练脚本具体改动对照

### `train_unet_proj_2d.py` vs `train_unet_2d.py`

```diff
- from datasets import raw_reader
+ from datasets import proj_reader

- train_dataset = raw_reader(paired_img_txt='./train_img.txt')
+ train_dataset = proj_reader(paired_img_txt='./train_proj_img.txt')

- valid_dataset = raw_reader(paired_img_txt='./valid_img.txt')
+ valid_dataset = proj_reader(paired_img_txt='./valid_proj_img.txt')

- train_loader = DataLoader(..., batch_size=12, ...)
+ train_loader = DataLoader(..., batch_size=32, ...)   # 输入小了 4 倍，可加大

- result_path = './runs/DenseUNet_PhaseCT/logs/'
- save_dir = './runs/DenseUNet_PhaseCT/checkpoints/'
+ result_path = './runs/DenseUNet_ProjCT/logs/'
+ save_dir = './runs/DenseUNet_ProjCT/checkpoints/'
```

其余代码（模型、loss、优化器、训练/验证循环）**完全不变**。

---

## 十、如果需要查看重建结果

投影域训练完成后，如果需要看到最终的**图像域指标**，还需要：

1. 将预测的投影数据 `proj_pred.raw` 通过**重建算法**（FBP / 滤波反投影）转为重建图像
2. 在重建图像上计算 PSNR/SSIM（HU 值）

当前项目中**没有包含重建算法代码**，如果后续需要，可以考虑：
- 使用 `scipy` 或 `astra-toolbox` 实现 FBP
- 或者用 `pytorch-fbp` 等库做可微重建
