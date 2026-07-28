# 测试脚本 `test.py` 说明

> 统一推理入口，支持 PhaseCT（图像域）和 ProjCT（投影域），替代原 AICT 的 `test_raw.py` 和 `test_proj_raw.py`。

---

## 1. 整体流程

```
加载 YAML 配置 ──→ 根据 model.name 构建网络 ──→ 加载 .dat 权重
                        │
                        ▼
            ┌─ phase? ─→ _infer_phase()
            └─ proj?  ─→ _infer_proj()
                              │
                              ▼
                   逐切片/视角推理 → 保存 input/pred/gt → 计算指标
```

## 2. 文件结构

```
test.py
├── main()                # 解析 CLI 参数，构建模型，分派到 _infer_phase / _infer_proj
├── _compute_metrics()    # 计算 MSE, RMSE, MAE, PSNR, SSIM
├── _infer_phase()        # 图像域推理（PhaseCT）
└── _infer_proj()         # 投影域推理（ProjCT）
```

## 3. CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config / -c` | 实验 YAML 配置（读取模型结构、归一化参数等） | 必填 |
| `--checkpoint` | 模型权重 `.dat` 文件 | 必填 |
| `--data-dir` | 数据根目录 | 必填 |
| `--output / -o` | 输出目录 | `./results/` |
| `--domain` | `phase` 或 `proj` | `phase` |
| `--patient` | 指定病人（空格分隔多个），不指定则用默认验证集 | 全部验证集 |
| `--gpu` | GPU 设备 ID | `0` |

示例：

```bash
# PhaseCT 推理（默认病人：17238, 18537）
python test.py --config experiments/denseunet_phasect.yaml \
    --checkpoint ./runs/DenseUNet_PhaseCT/checkpoints/model_at_epoch_200.dat \
    --data-dir /root/autodl-tmp/联影双能相位数据2080 \
    --output ./results/

# ProjCT 推理
python test.py --config experiments/denseunet_projct.yaml \
    --checkpoint ./runs/DenseUNet_ProjCT/checkpoints/model_at_epoch_50.dat \
    --data-dir /root/autodl-tmp/联影双能相位数据2080 \
    --output ./results/ \
    --domain proj

# 指定特定病人
python test.py ... --patient "patient_A" "patient_B"

# 指定 GPU
python test.py ... --gpu 1
```

## 4. PhaseCT 推理（`_infer_phase`）

### 4.1 数据读取

从 `--data-dir/{patient}/` 下读取两个体数据文件：

| 文件 | 用途 | Shape |
|------|------|-------|
| `rec_pbi_fs_dec_blur_phase_100000.raw` | 模型输入（带伪影） | `(160, 512, 512)` |
| `rec_no_pbi.raw` | Ground Truth | `(160, 512, 512)` |

### 4.2 推理循环

跳过首尾切片（索引 0 和 159），对第 1~158 张切片逐张推理：

```
for i in range(1, 159):
    s = vol[i]                         # (512, 512)，原始 HU 值
    s = (s - hu_min) / hu_range        # 归一化到 [0,1]，hu_min/hu_max 从 YAML 读取
    s = clip(s, 0, 1)                  # 截断越界值
    t = tensor(s)[None, None, ...]     # 扩展为 (1,1,512,512)
    p = model(t)                       # 网络推理
    p = p * hu_range + hu_min          # 反归一化回 HU 值
    preds.append(p)
```

- 归一化和反归一化公式与训练完全一致
- `hu_min` / `hu_max` 从 `--config` 的 YAML 中 `data.hu_min` / `data.hu_max` 读取，默认 `-1000` / `1000`

### 4.3 保存结果

输出到 `{output}/phase/{patient}/`：

| 文件 | 内容 | Shape |
|------|------|-------|
| `input.raw` | 输入体数据（第 1~158 切片） | `(158, 512, 512)` |
| `pred.raw` | 模型预测结果 | `(158, 512, 512)` |
| `gt.raw` | Ground Truth（第 1~158 切片） | `(158, 512, 512)` |

## 5. ProjCT 推理（`_infer_proj`）

### 5.1 数据读取

| 文件 | 用途 | Shape |
|------|------|-------|
| `proj_pbi_fs_dec_blur_phase_100000.raw` | 模型输入（带伪影投影） | `(720, 128, 512)` |
| `proj_no_pbi.raw` | Ground Truth | `(720, 128, 512)` |

### 5.2 推理循环

遍历全部 720 个视角：

```
for i in range(720):
    s = vol[i]                         # (128, 512)
    if crop_rows > 0:
        s = s[crop_rows:-crop_rows]    # 上下裁剪黑边 (128→112)
    s = clip(s, 0, norm_clip_max)      # 截断到 [0, norm_clip_max]
    s = s / norm_clip_max              # 归一化到 [0,1]
    t = tensor(s)[None, None, ...]     # (1,1, target_rows, 512)
    p = model(t)
    p = p * norm_clip_max              # 反归一化
    preds.append(p)
```

- `crop_rows` 和 `norm_clip_max` 从 YAML 读取，与训练完全一致
- GT 和输入也会按相同 `crop_rows` 裁剪后保存，保证与 prediction 维度一致

### 5.3 保存结果

输出到 `{output}/proj/{patient}/`：

| 文件 | 内容 | Shape |
|------|------|-------|
| `input.raw` | 裁剪后的输入体数据 | `(720, target_rows, 512)` |
| `pred.raw` | 模型预测结果 | `(720, target_rows, 512)` |
| `gt.raw` | 裁剪后的 Ground Truth | `(720, target_rows, 512)` |

其中 `target_rows = 128 - 2 * crop_rows`（如 `crop_rows=8` 时为 112）。

## 6. 指标计算

`_compute_metrics(pred, gt, tag)` 函数计算五个指标：

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| **MSE** | `mean((pred - gt)²)` | 均方误差 |
| **RMSE** | `sqrt(MSE)` | 均方根误差 |
| **MAE** | `mean(|pred - gt|)` | 平均绝对误差 |
| **PSNR** | `skimage.metrics.peak_signal_noise_ratio` | 峰值信噪比，`data_range` 取 GT 的动态范围 |
| **SSIM** | `skimage.metrics.structural_similarity` 逐切片计算后取均值 | 结构相似性 |

每个病人输出两组指标：

```
── Metrics for 17238_33314... ──
  input vs gt :  MSE: 0.001234  RMSE: 0.035128  MAE: 0.021456  PSNR: 29.08  SSIM: 0.8912
  pred  vs gt :  MSE: 0.000456  RMSE: 0.021354  MAE: 0.012345  PSNR: 33.41  SSIM: 0.9421
```

- **input vs gt**：输入数据与 Ground Truth 的差距（基线，代表去噪前的质量）
- **pred vs gt**：模型预测与 Ground Truth 的差距（代表去噪后的质量提升）

两个对比可以直观看出模型的去噪效果。

## 7. 与 YAML 配置的联动

`test.py` 从 YAML 配置中读取以下参数，确保测试与训练一致：

| YAML 字段 | 在测试中的用途 | 影响 |
|-----------|---------------|------|
| `model.name` | 选择网络结构（DenseUNet2d / UNet2d） | 模型架构 |
| `model.in_chl` | 输入通道数 | |
| `model.out_chl` | 输出通道数 | |
| `model.model_chl` | 基础通道数 | |
| `data.hu_min` (raw) | PhaseCT 归一化下界 | 与训练的归一化一致 |
| `data.hu_max` (raw) | PhaseCT 归一化上界 | 与训练的归一化一致 |
| `data.norm_clip_max` (proj) | ProjCT 投影值截断上限 | 与训练的归一化一致 |
| `data.crop_rows` (proj) | ProjCT 上下裁剪行数 | 保证输入/输出/GT 维度与训练一致 |

## 8. 参考

- 原版 PhaseCT 测试：`C:\imageReconstruction\AICT\test_raw.py`
- 原版 ProjCT 测试：`C:\imageReconstruction\AICT\test_proj_raw.py`
- 训练配置：`TRAINING_GUIDE.md §5.1`
- 数据格式：`DATASET_FORMAT.md`
