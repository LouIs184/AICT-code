# TMCode PBI 去伪影功能 — YAML 配置指南

> 本项目移植了同学 `TMCode` 的两项能力：**伪影残差模型输出**（`model.artifact_output`）和 **PBI 去伪影复合损失**（`loss.l1_weight / artifact_weight / fft_amp_weight / angular_weight / sobel_weight`）。
> 这篇文档只讲怎么在 YAML 里开启和调参。实现细节见 `docs/修改日志/5-TMCode_PBI_损失与模型输出.md`，差异背景见 `docs/差异分析.md`。

---

## 1. 快速启用（推荐配置）

直接用已写好的配置跑即可：

```bash
python train.py --config experiments/denseunet_projct_TM.yaml
```

它等价于下面这份配置（**复现同学损失组合**）：

```yaml
model:
  name: DenseUNet2d
  artifact_output: true          # ← 关键开关：伪影残差输出 + 恒等起点

loss:
  mse_weight: 0.0                # 换成复合损失，MSE 关掉
  adv_weight: 0.0                # 关 GAN（默认 0.0，>0 开启）
  l1_weight: 0.3                 # 干净图 L1
  artifact_weight: 0.1           # 伪影残差 L1（真实伪影 = 输入 - GT）
  ssim_weight: 0.8               # 1 - SSIM
  fft_amp_weight: 0.8            # FFT 幅值 L1（压周期条纹）
  angular_weight: 0.8            # 沿投影轴向梯度 L1
  sobel_weight: 0.3              # 多向 Sobel 梯度 L1
```

---

## 2. 两个开关分别管什么

### 2.1 `model.artifact_output`（模型侧，总开关）

| 值 | 效果 |
|---|---|
| `false`（默认） | DenseUNet2d 保持原行为，输出 `F.relu(x + conv6)`（直接回归） |
| `true` | 模型改为**预测伪影**：输出 `(clean, artifact)`，`clean = x - artifact`；输出层近零初始化，训练起点≈恒等映射 |

**注意**：`artifact_output: true` 时模型返回元组，训练/验证/推理脚本都会自动取 `clean`（已兼容），但 **`loss.artifact_weight` 必须 > 0 才有意义**（见 2.2）。

### 2.2 损失权重（损失侧）

| 字段 | 同学取值 | 作用 | 依赖 |
|---|---|---|---|
| `l1_weight` | 0.3 | 干净图与 GT 的 L1 | 无 |
| `artifact_weight` | 0.1 | 预测伪影与真实伪影的 L1 | **需 `artifact_output: true`**（否则该分项跳过） |
| `ssim_weight` | 0.8 | `1 - SSIM`（预测先 clamp 到 [0,1]） | 无 |
| `fft_amp_weight` | 0.8 | FFT 幅值域 L1（`log1p(|FFT|)`） | 无 |
| `angular_weight` | 0.8 | 沿最后一个维度（投影轴向）梯度 L1 | 无 |
| `sobel_weight` | 0.3 | 0°/90°/45°/135° 多向 Sobel 梯度 L1 | 无 |

> 权重都设 0 就等价于没启用这项；`mse_weight` 等原有损失仍然可用，可叠加。

---

## 3. 常用配置组合

### 组合 A：完全复现同学（新配置已内置）

上面第 1 节的配置即可。特征：`mse_weight=0`，只用复合损失。

### 组合 B：保留 MSE 保底

```yaml
loss:
  mse_weight: 0.5       # 保留 MSE 保底
  l1_weight: 0.3
  artifact_weight: 0.1
  ssim_weight: 0.8
  fft_amp_weight: 0.8
  angular_weight: 0.8
  sobel_weight: 0.3
```

### 组合 C：只换损失、不改模型输出

```yaml
model:
  artifact_output: false    # 保持原直接回归输出
loss:
  mse_weight: 0.0
  l1_weight: 1.0            # 用 L1 替代 MSE
  ssim_weight: 1.0
  fft_amp_weight: 1.0
  angular_weight: 1.0
  sobel_weight: 1.0
  # artifact_weight 不设（模型不输出伪影，该项自动跳过）
```

> 注意：组合 C 没有恒等起点和伪影监督，效果会打折，只建议做消融对比用。

---

## 4. 配套建议（可选）

| 项 | 建议 | 说明 |
|---|---|---|
| 调度器 | 换成 `CosineAnnealingLR` | 同学用的是 CosineAnnealing(T_max=150)，你的 `StepLR(step=2,γ=0.95)` 也可用，两者差异不大 |
| 梯度裁剪 | 建议加 | 同学用 `clip_grad_value_(0.01)`。FFT/梯度类损失梯度偏大，若训练震荡就在 trainer 里补一行 |
| 数据归一化 | 保持 `norm_clip_max: 0.28` | 同学是 /0.3，很接近，不必改 |

---

## 5. 怎么确认功能真的生效了

训练第 1 个 epoch 观察：

1. **TensorBoard / 控制台出现新分项**：`loss/train_l1`、`loss/train_artifact`、`loss/train_fft_amp`、`loss/train_angular`、`loss/train_sobel`，控制台打印 `L1 / ARTIFACT / SSIM / FFT_AMP / ANGULAR / SOBEL`。
2. **恒等起点**：第 1 个 epoch 的 `artifact` 损失应接近初始值（`≈ L1(0, lq-gt)`），`l1` 损失接近输入与 GT 的初始差异——说明模型确实从"输出≈输入"开始。
3. **可视化**：`train img/predict` 显示的是**校正后的干净图**，而不是伪影图或灰图。

---

## 6. 常见坑

1. **改了 `artifact_output: true` 但忘了设 `artifact_weight`**：伪影残差分项不会报错，只是被跳过（权重 0），恒等起点白白浪费一半效果。
2. **`mse_weight` 没关**：复合损失已经很大，再叠 MSE 会让整体损失比例失衡。复现同学时应设 `mse_weight: 0`（组合 B 除外）。
3. **GAN 配置冲突**：`adv_weight > 0` 且 `artifact_output: true` 暂不支持（`_train_gan_step` 假设单张输出）。用新功能时保持 `adv_weight: 0`（trainer 已加保护：二者同开会直接报错提示，不会再晦涩崩溃）。
4. **旧 checkpoint 兼容**：`artifact_output` 不改变权重结构，旧权重可正常加载，但加载后模型按配置的 `artifact_output` 决定输出方式。

---

## 7. 相关文件

- 配置示例：`experiments/denseunet_projct_TM.yaml`
- 损失实现：`src/losses/sino.py`
- 模型实现：`src/models/unet.py`（DenseUNet2d）
- 训练器：`src/trainer.py`
- 方案文档：`docs/AICTVer2_添加TMCode_损失与模型输出_改造方案.md`
- 修改日志：`docs/修改日志/5-TMCode_PBI_损失与模型输出.md`
