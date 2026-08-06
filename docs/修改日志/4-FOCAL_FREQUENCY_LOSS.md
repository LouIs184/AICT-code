# Focal Frequency Loss（FFL，ICCV 2021）接入修改日志

> 论文：Focal Frequency Loss for Image Reconstruction and Synthesis（ICCV 2021）
> <https://arxiv.org/pdf/2012.12821.pdf>
> 参考源码：官方实现 `limingcv/Focal-Frequency-Loss`（<https://github.com/limingcv/Focal-Frequency-Loss>），本改动把官方算法移植进 `src/losses/`。

---

## 一、改造目标

给项目增加**频域损失** `FocalFrequencyLoss`：

- 在 2D DFT 频域比较预测与真值的频谱，对误差大的"难"频率分量动态赋予更高权重（频谱权重矩阵），引导模型优先修复频域难点；
- 与现有空间域损失（MSE / Charbonnier / FocalMSE）**互补**，可按论文建议组合使用；
- 完全通过 YAML 开关，不开时行为与原来完全一致，向后兼容。

**改动范围**：5 个文件（新增 1 个）。

---

## 二、具体改动

### 2.1 `src/losses/focal_frequency.py`（新增，核心）

`FocalFrequencyLoss`，算法与官方实现（`focal_frequency_loss/focal_frequency_loss.py`）完全一致，仅做两处适配：

1. 项目环境 `torch>=2.0`，直接使用 `torch.fft.fft2`，移除官方代码里的 `IS_HIGH_VERSION` 版本分支判断；
2. 按项目风格改写中文 docstring / 注释。

保留官方全部构造参数（默认值一致）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `loss_weight` | `1.0` | 整体缩放系数（本项目通过 YAML 的 `ffl_weight` 在 trainer 外部加权，构造时保持 `1.0`） |
| `alpha` | `1.0` | 频谱权重矩阵缩放指数 |
| `patch_factor` | `1` | 图像切块因子，>1 为 patch-based FFL |
| `ave_spectrum` | `False` | 是否使用 minibatch 平均频谱 |
| `log_matrix` | `False` | 是否对频谱权重矩阵取对数调整 |
| `batch_matrix` | `False` | 是否用 batch 级统计归一化权重矩阵 |

**算法流程**（`tensor2freq` → `loss_formulation`）：

```python
freq = torch.fft.fft2(y, norm="ortho")     # 切块后 2D FFT
freq = torch.stack([freq.real, freq.imag], -1)  # 实部/虚部分别存最后一维

# loss_formulation：在线动态计算频谱权重矩阵
matrix_tmp = torch.sqrt(((recon - real) ** 2)[..., 0] + ((recon - real) ** 2)[..., 1]) ** alpha
matrix_tmp = torch.log(matrix_tmp + 1.0) if log_matrix else matrix_tmp
matrix_tmp = matrix_tmp / matrix_tmp.max()            # batch_matrix: batch 级
matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]  # 否则逐样本
matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
weight_matrix = torch.clamp(matrix_tmp, 0.0, 1.0).clone().detach()  # 关键：detach，不参与梯度

# 频谱（平方）欧氏距离 × 权重矩阵（Hadamard 积）后取均值
tmp = (recon_freq - real_freq) ** 2
freq_distance = tmp[..., 0] + tmp[..., 1]
loss = (weight_matrix * freq_distance).mean()
```

**约定**：权重矩阵必须 `detach()`；归一化到 `[0, 1]`，loss 量级由频谱欧氏距离决定，不会漂移。

### 2.2 `src/losses/__init__.py`

导出 `FocalFrequencyLoss`，加入 `__all__`。

### 2.3 `src/config.py` — `LossConfig` 新增 6 个字段

```python
# ---- Focal Frequency Loss（FFL，ICCV 2021，见 src/losses/focal_frequency.py）----
ffl_weight: float = 0.0         # >0 时启用 FFL（与 mse_weight 组合，频域互补）
ffl_alpha: float = 1.0          # 频谱权重矩阵缩放指数 alpha
ffl_patch_factor: int = 1       # 图像切块因子（1=不切块，>1 为 patch-based FFL）
ffl_ave_spectrum: bool = False  # 使用 minibatch 平均频谱
ffl_log_matrix: bool = False    # 对频谱权重矩阵取对数调整
ffl_batch_matrix: bool = False  # 用 batch 级统计归一化权重矩阵
```

### 2.4 `src/trainer.py`

- `import` 引入 `FocalFrequencyLoss`；
- `_build_losses()`：`ffl_weight>0` 时实例化 `self.criterion_ffl`（传入 `alpha` / `patch_factor` / `ave_spectrum` / `log_matrix` / `batch_matrix`）；
- `_compute_loss()`：`ffl_weight>0` 时把 ffl 项加入总损失，并写入 `losses["ffl"]`。

调用点（`_train_standard_step` / `_train_gan_step` / 深度监督）**无需改动**——FFL 只用 `pred` / `target`，不依赖网络输入，GAN / Deep Supervision 分支自动生效。

### 2.5 `experiments/denseunet_projct.yaml`

按 FocalMSE 的写法在 `loss:` 段追加注释示例配置（默认全关闭）。

---

## 三、如何使用

在实验 YAML（如 `experiments/denseunet_projct.yaml`）的 `loss:` 段加入 ffl 字段即可，**不开则行为与原来完全一致**：

```yaml
loss:
  mse_weight: 1.0          # 建议保留 MSE 保底，与 FFL 组合
  ffl_weight: 1.0          # FFL 权重，>0 时启用；起点建议 0.1 ~ 1.0
  ffl_alpha: 1.0           # 频谱权重矩阵缩放指数
  ffl_patch_factor: 1      # 1=不切块（默认）
  ffl_ave_spectrum: false  # minibatch 平均频谱
  ffl_log_matrix: false    # 对权重矩阵取对数
  ffl_batch_matrix: false  # batch 级统计归一化
```

训练时总损失 = `mse_weight * MSE + ffl_weight * FFL(+ 其他已配置项)`。TensorBoard 中会多出 `loss/train_ffl` 分项，控制台显示 `FFL`。

**推荐实验顺序**：
1. 基线：`mse_weight: 1.0`，不开 ffl —— 记录整体/关键区指标；
2. `mse_weight: 1.0` + `ffl_weight: 0.1 ~ 1.0` —— 对比频域损失是否提升高频细节 / 伪影区；
3. 若想更聚焦高频，可试 `ffl_log_matrix: true`（对数压低权重差异）或 `ffl_alpha` 微调。

---

## 四、改动汇总

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `src/losses/focal_frequency.py` | ✅ 新增 | `FocalFrequencyLoss`（官方算法移植，torch>=2.0 适配） |
| `src/losses/__init__.py` | ✅ 修改 | 导出 `FocalFrequencyLoss` |
| `src/config.py` | ✅ 新增字段 | `LossConfig` 新增 `ffl_*` 6 个字段 |
| `src/trainer.py` | ✅ 修改 | `_build_losses()` 构建、`_compute_loss()` 计算并记录 `ffl` 分项 |
| `experiments/denseunet_projct.yaml` | ✅ 新增配置 | 追加 FFL 注释示例 |

**向后兼容性**：`ffl_weight=0.0` 是默认值，不启用 FFL，行为与修改前完全一致。`test.py` / `infer.py` 只算指标不算 loss，无需改动。

---

## 五、验证

跑通后确认两件事：
- 训练日志 / TensorBoard 出现 `loss/train_ffl` 分项；
- `patch_factor > 1` 时必须保证图像高宽能被整除（本项目投影域裁剪后尺寸均满足）；不满足会报断言错误。

---

## 六、注意事项

1. **保留少量普通 MSE 保底**：FFL 是频域权衡损失，建议与 `mse_weight: 1.0` 组合，不要全替换。
2. **权重矩阵已 detach**：不参与梯度回传，避免等价于更高次幂损失、梯度放大。
3. **默认参数即可起步**：`patch_factor` / `ave_spectrum` / `batch_matrix` / `log_matrix` 一般保持默认，先只调 `ffl_weight` 和 `ffl_alpha`。
4. **不要只盯整体 MSE 选 checkpoint**：FFL 牺牲部分整体指标换频域/关键区质量，请结合 PSNR/SSIM 与可视化残差图综合判断。
