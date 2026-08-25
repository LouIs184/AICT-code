# 移植 TMCode「PBI 去伪影损失 + 伪影残差模型输出」修改日志

> 背景：同学项目 `TMCode`（ResidualFBPConvNet + 六项复合损失）在投影域 PBI 条纹校正上效果明显优于本项目。
> 本次只移植其中**两项**：1. 损失函数；2. 模型输出（伪影残差 + 恒等起点）。
> 方案文档：`docs/AICTVer2_添加TMCode_损失与模型输出_改造方案.md`；差异对比：`docs/差异分析.md`。

---

## 一、改造目标

1. **损失函数**：从纯 MSE（`mse_weight=1.0`，其余全 0）换成同学的六项复合损失：
   `0.3·L1 + 0.1·伪影残差L1 + 0.8·(1-SSIM) + 0.8·FFT幅值 + 0.8·角度梯度 + 0.3·多向Sobel`；
2. **模型输出**：DenseUNet2d 改为**预测伪影**，`clean = input - predicted_artifact`，输出层近零初始化 → **恒等起点**；
3. 完全通过 YAML 开关，旧配置行为不变，向后兼容。

**改动范围**：8 个文件（新增 2 个）。

---

## 二、具体改动

### 2.1 `src/losses/sino.py`（新增，核心）

移植同学 `TMCode/train_fbpconvnet_sino.py` 的 3 个损失，包成 `nn.Module`：

| 类 | 对应同学实现 | 作用 |
|---|---|---|
| `MultiDirectionalSobelLoss` | `MultiDirectionalSobelLoss` | 0°/90°/45°/135° 四方向 Sobel 梯度 L1 |
| `FFTAmplitudeLoss` | `log_fft_loss` | FFT 幅值域 `log1p(|FFT|)` 的 L1，压制周期性条纹 |
| `AngularGradLoss` | `angular_grad_loss` | 沿最后一个维度（投影轴向）一维梯度 L1 |

要点：
- Sobel 4 个 kernel 注册为 `register_buffer`（自动随设备迁移）；
- FFT 用 `torch.fft.fft2(..., norm="backward")`（torch ≥ 1.7）。

### 2.2 `src/losses/__init__.py`

导出 `MultiDirectionalSobelLoss` / `FFTAmplitudeLoss` / `AngularGradLoss`，加入 `__all__`。

### 2.3 `src/config.py`

- `ModelConfig` 新增 `artifact_output: bool = False`（true 时模型输出 `(clean, artifact)`）；
- `LossConfig` 新增 5 个权重字段：

```python
# ---- TMCode PBI 去伪影复合损失（移植自 TMCode/train_fbpconvnet_sino.py）----
l1_weight: float = 0.0          # 干净图 L1 权重（同学用 0.3）
artifact_weight: float = 0.0    # 伪影残差 L1 权重（需 model.artifact_output=true，同学用 0.1）
fft_amp_weight: float = 0.0     # FFT 幅值 L1 权重（同学用 0.8）
angular_weight: float = 0.0     # 沿末维梯度 L1 权重（同学用 0.8）
sobel_weight: float = 0.0       # 多向 Sobel 梯度 L1 权重（同学用 0.3）
```

### 2.4 `src/models/unet.py`

- `UNet2d` / `UNet2dDeepSuperviser` / `DenseUNet2d` 的 `__init__` 都加 `artifact_output=False` 参数（保持 trainer 调用统一）；
- **只有 `DenseUNet2d` 实现**该逻辑：
  - forward：`artifact = end(up1)`（不加 ReLU），`clean = x - artifact`，返回 `(clean, artifact)`；
  - 构造时若 `artifact_output=True`，把 `self.end[0].conv` 近零初始化（`std=0.001`）→ 训练起点 `artifact≈0`、`clean≈x`（恒等映射）。

### 2.5 `src/trainer.py`

- `import` 引入 3 个新损失；
- `_build_model()`：构造模型时透传 `artifact_output=mc.artifact_output`；
- `_build_losses()`：新增 `criterion_l1` / `criterion_fft_amp` / `criterion_angular` / `criterion_sobel`；
- `_compute_loss()`：签名改为 `(predict, target, pred_artifact=None, gt_artifact=None)`（原未使用的 `model_out` 参数移除），在 MSE 之后插入 l1 / artifact / fft_amp / angular / sobel 五项；SSIM 项改为先 `predict.clamp(0, 1)` 再算（对齐同学实现）；
- `_train_standard_step()`：`model(ldct)` 输出为元组且 `artifact_output=True` 时，拆出 `(pred_clean, pred_artifact)`，`gt_artifact = ldct - rdct`，把伪影残差传入 `_compute_loss`；图片日志用 `pred_clean`；
- `_validate()`：`artifact_output` 时取 `predict[0]`（clean）而非 `predict[-1]`。

### 2.6 `experiments/denseunet_projct_TM.yaml`（新增）

```yaml
model:
  artifact_output: true

loss:
  mse_weight: 0.0      # 换用复合损失
  adv_weight: 0.0      # 关 GAN（默认 0.0，>0 开启）
  l1_weight: 0.3
  artifact_weight: 0.1
  ssim_weight: 0.8
  fft_amp_weight: 0.8
  angular_weight: 0.8
  sobel_weight: 0.3
```

### 2.7 `test.py` / `infer.py`

4 处推理调用点加元组兼容保护：`model()` 返回 `(clean, artifact)` 时取 `p[0]`，防止 `artifact_output` 模型推理时对元组调用 `.cpu().numpy()` 报错。

---

## 三、如何使用

```bash
python train.py --config experiments/denseunet_projct_TM.yaml
```

**旧配置不受影响**：`artifact_output` 默认 `False`、5 个新权重默认 `0`，不写这些字段时行为与修改前完全一致；checkpoint 兼容（权重结构未变）。

---

## 四、改动汇总

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `src/losses/sino.py` | ✅ 新增 | 3 个损失类：`MultiDirectionalSobelLoss` / `FFTAmplitudeLoss` / `AngularGradLoss` |
| `experiments/denseunet_projct_TM.yaml` | ✅ 新增 | 启用 `artifact_output` + 同学损失权重 |
| `src/losses/__init__.py` | ✅ 修改 | 导出 3 个新损失 |
| `src/config.py` | ✅ 修改 | `ModelConfig.artifact_output`；`LossConfig` 新增 5 个权重 |
| `src/models/unet.py` | ✅ 修改 | 3 个模型加 `artifact_output` 参数；DenseUNet2d 伪影输出 + 恒等起点初始化 |
| `src/trainer.py` | ✅ 修改 | `_build_model` / `_build_losses` / `_compute_loss` / `_train_standard_step` / `_validate` |
| `test.py` | ✅ 修改 | 推理调用点元组兼容保护 |
| `infer.py` | ✅ 修改 | 推理调用点元组兼容保护 |

---

## 五、验证

跑通后确认：
1. TensorBoard 出现 `loss/train_l1` / `loss/train_artifact` / `loss/train_fft_amp` / `loss/train_angular` / `loss/train_sobel` 等分项，控制台打印 `L1 / ARTIFACT / SSIM / FFT_AMP / ANGULAR / SOBEL`；
2. 第 1 个 epoch 的 `artifact` 损失接近初始值、`l1` 损失接近输入与 GT 的初始 L1 → 恒等起点生效；
3. `train img/predict` 显示的是**校正图（clean）**，不是伪影图。

---

## 六、注意事项

1. **元组输出顺序**：`(clean, artifact)`，clean 在**第 0 个**。`_validate` / `_log_images` / 推理脚本取值处都要按 `artifact_output` 分支，否则会把伪影图当结果。
2. **artifact 不加 ReLU**：伪影可正可负，`conv6` 输出不要过 `F.relu`。
3. **恒等初始化只动输出层** `self.end[0].conv`，不要动网络其它层。
4. **GAN + `artifact_output` 暂不支持**：`_train_gan_step` 假设单张输出；新配置 `adv_weight=0` 不受影响，若开启 GAN 需另行适配。
5. **SSIM clamp**：SSIM 项改为先 clamp `[0,1]` 再算（对齐同学），只在 `ssim_weight>0` 时生效。
6. **本次未移植** InstanceNorm 与梯度裁剪（见 `docs/差异分析.md` 3.3 / 3.4）。若训练震荡/发散，补一行即可：`torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)`。
