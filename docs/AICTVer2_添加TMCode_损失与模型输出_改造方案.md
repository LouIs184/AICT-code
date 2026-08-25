# AICTVer2 移植 TMCode「损失函数 + 模型输出」改造方案

> 目标：把 `TMCode`（同学项目）里 **1. 损失函数** 和 **2. 模型输出** 两项移植到 AICTVer2。
> 对应文档 `docs/AICTVer2_vs_TMCode_差异分析.md` 的第 3.1、3.2 节。
> 只做这两项，**不做** InstanceNorm（第 3.3 节）和梯度裁剪（第 3.4 节）。
>
> 背景：同学损失 = `0.3·L1 + 0.1·伪影L1 + 0.8·(1-SSIM) + 0.8·FFT幅值 + 0.8·角度梯度 + 0.3·多向Sobel`；
> 模型输出 = `clean = input - predicted_artifact`（预测伪影，输出层近零初始化 → 恒等起点）。

---

## 1. 改动总览

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 1 | `src/losses/sino.py` | **新增** | 移植同学的 3 个损失类 |
| 2 | `src/losses/__init__.py` | 导出新损失 | |
| 3 | `src/config.py` | LossConfig + ModelConfig 加字段 | 后向兼容，旧 yaml 不受影响 |
| 4 | `src/models/unet.py` | DenseUNet2d 加 `artifact_output` | 伪影残差输出 + 恒等起点 |
| 5 | `src/trainer.py` | `_build_model` / `_build_losses` / `_compute_loss` / `_train_standard_step` / `_validate` | 核心改动 |
| 6 | `experiments/denseunet_projct_TM.yaml` | **新增**配置文件 | 启用新损失 + 伪影输出 |

`src/models/__init__.py` **不需要改**（类名 `DenseUNet2d` 不变）。

---

## 2. 改动 1：新增 `src/losses/sino.py`

把同学 `TMCode/train_fbpconvnet_sino.py` 里的 3 个损失函数移植成模块（代码原样照搬，包成 nn.Module）：

```python
"""
TMCode PBI 去伪影专用损失（投影域/正弦图）。

移植自 TMCode/train_fbpconvnet_sino.py 的：
- MultiDirectionalSobelLoss：0°/90°/45°/135° 多向梯度 L1
- FFTAmplitudeLoss       ：FFT 幅值域 log1p L1（对应同学 log_fft_loss）
- AngularGradLoss        ：沿最后一个维度（投影轴向）一维梯度 L1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDirectionalSobelLoss(nn.Module):
    """0°/90°/45°/135° 四方向 Sobel 梯度 L1。"""

    def __init__(self):
        super().__init__()
        k0 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        k90 = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        k45 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
        k135 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32)
        kernels = torch.stack([k0, k90, k45, k135]).unsqueeze(1)  # (4,1,3,3)
        self.register_buffer("kernels", kernels)

    def forward(self, pred, gt):
        pred_grad = F.conv2d(pred, self.kernels, padding=1)
        gt_grad = F.conv2d(gt, self.kernels, padding=1)
        return F.l1_loss(pred_grad, gt_grad)


class FFTAmplitudeLoss(nn.Module):
    """频域幅值 L1：log1p(|FFT(pred)|) 与 log1p(|FFT(gt)|) 的 L1，压制周期性条纹。"""

    def forward(self, pred, target):
        fft_pred = torch.fft.fft2(pred, norm="backward")
        fft_tar = torch.fft.fft2(target, norm="backward")
        amp_pred = torch.log1p(torch.abs(fft_pred))
        amp_tar = torch.log1p(torch.abs(fft_tar))
        return torch.mean(torch.abs(amp_pred - amp_tar))


class AngularGradLoss(nn.Module):
    """沿最后一个维度（投影/视角轴向）的一维梯度 L1，直接约束条纹方向。"""

    def forward(self, pred, gt):
        pred_grad = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_grad = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        return F.l1_loss(pred_grad, gt_grad)
```

> 注意：`torch.fft.fft2` 需要 torch ≥ 1.7。

---

## 3. 改动 2：`src/losses/__init__.py` 导出

在现有内容后追加：

```python
from .sino import MultiDirectionalSobelLoss, FFTAmplitudeLoss, AngularGradLoss
```

并在 `__all__` 里加入：

```python
    "MultiDirectionalSobelLoss",
    "FFTAmplitudeLoss",
    "AngularGradLoss",
```

---

## 4. 改动 3：`src/config.py` 增加字段

### 4.1 `LossConfig`（第 55-78 行附近）末尾追加：

```python
    # ---- TMCode PBI 去伪影复合损失（移植自 TMCode/train_fbpconvnet_sino.py）----
    l1_weight: float = 0.0          # 干净图 L1 权重（同学用 0.3）
    artifact_weight: float = 0.0    # 伪影残差 L1 权重（需 model.artifact_output=true，同学用 0.1）
    fft_amp_weight: float = 0.0     # FFT 幅值 L1 权重（同学用 0.8）
    angular_weight: float = 0.0     # 沿末维梯度 L1 权重（同学用 0.8）
    sobel_weight: float = 0.0       # 多向 Sobel 梯度 L1 权重（同学用 0.3）
```

### 4.2 `ModelConfig`（第 46-52 行）追加：

```python
    artifact_output: bool = False   # true: 输出 (clean, artifact)，clean = x - artifact（恒等起点）
```

> 后向兼容：`load_config`（`config.py:160`）对未知 key 静默忽略、已知字段用默认值。旧 yaml 不写这些字段时行为不变。

---

## 5. 改动 4：`src/models/unet.py` — DenseUNet2d 加 `artifact_output`

### 5.1 给三个模型类的 `__init__` 都加参数（保持 trainer 调用统一）

`UNet2d`（`:24`）、`UNet2dDeepSuperviser`（`:76`）、`DenseUNet2d`（`:128`）的签名改为：

```python
    def __init__(self, in_chl=1, out_chl=1, model_chl=32, artifact_output=False):
```

UNet2d / DeepSuperviser 只是**接受参数**、forward 保持原样；**只有 DenseUNet2d 实现**伪影输出逻辑（当前用的就是它）。

### 5.2 `DenseUNet2d.forward` 修改

把 `src/models/unet.py:154-166` 的 forward 改成：

```python
    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        if self.artifact_output:
            artifact = conv6          # 预测伪影（不加 ReLU，伪影可正可负）
            clean = x - artifact      # 校正图 = 输入 - 伪影
            return clean, artifact
        return F.relu(x + conv6)      # 原行为
```

### 5.3 恒等起点初始化

在 `DenseUNet2d.__init__` 里 `self.end` 定义之后加（`unet.py:149-152` 附近）：

```python
        if artifact_output:
            # 输出层近零初始化 → 训练一开始 artifact≈0，clean≈x（恒等起点）
            conv = self.end[0].conv
            nn.init.normal_(conv.weight, mean=0.0, std=0.001)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)
```

> 说明：`self.end` 是 `nn.Sequential(ConvBnRelu2d(...))`，真正的卷积层在 `self.end[0].conv`。
> 注意先存 `self.artifact_output = artifact_output`。

---

## 6. 改动 5：`src/trainer.py`（核心）

### 6.1 导入（第 30 行）

```python
from src.losses import (
    CharbonnierLoss, SSIM, vgg_feature_extractor, vgg_loss_calc, TVLoss,
    FocalMSE, FocalFrequencyLoss,
    MultiDirectionalSobelLoss, FFTAmplitudeLoss, AngularGradLoss,
)
```

### 6.2 `_build_model`（第 180 行）传参

```python
        model = model_cls(in_chl=mc.in_chl, out_chl=mc.out_chl, model_chl=mc.model_chl,
                          artifact_output=mc.artifact_output)
```

### 6.3 `_build_losses`（第 193-198 行）追加

```python
        self.criterion_l1 = nn.L1Loss()
        self.criterion_fft_amp = FFTAmplitudeLoss().to(self.device)
        self.criterion_angular = AngularGradLoss().to(self.device)
        self.criterion_sobel = MultiDirectionalSobelLoss().to(self.device)
```

### 6.4 `_compute_loss`（第 266-310 行）

签名改为 `def _compute_loss(self, predict, target, pred_artifact=None, gt_artifact=None):`
（原 `model_out=None` 参数从未被使用，可安全替换）。

在 `if lc.mse_weight > 0:` 块**之后**、`if lc.charbonnier_weight > 0:` 之前插入：

```python
        # ---- TMCode 移植：L1 / 伪影残差 L1 / FFT 幅值 / 角度梯度 / 多向 Sobel ----
        if lc.l1_weight > 0:
            l_l1 = self.criterion_l1(predict, target)
            losses["l1"] = l_l1
            loss += lc.l1_weight * l_l1

        if lc.artifact_weight > 0 and pred_artifact is not None and gt_artifact is not None:
            l_art = self.criterion_l1(pred_artifact, gt_artifact)
            losses["artifact"] = l_art
            loss += lc.artifact_weight * l_art

        if lc.fft_amp_weight > 0:
            l_fft = self.criterion_fft_amp(predict, target)
            losses["fft_amp"] = l_fft
            loss += lc.fft_amp_weight * l_fft

        if lc.angular_weight > 0:
            l_ang = self.criterion_angular(predict, target)
            losses["angular"] = l_ang
            loss += lc.angular_weight * l_ang

        if lc.sobel_weight > 0:
            l_sob = self.criterion_sobel(predict, target)
            losses["sobel"] = l_sob
            loss += lc.sobel_weight * l_sob
```

同时把现有 SSIM 项（第 282-286 行）改成与同学一致、先 clamp 到 [0,1]：

```python
        if lc.ssim_weight > 0:
            l_ssim = 1 - self.criterion_ssim(predict.clamp(0, 1), target)
            losses["ssim"] = l_ssim
            loss += lc.ssim_weight * l_ssim
```

> 注意：clamp 只在 `ssim_weight > 0` 时生效，不影响其它配置。若想严格保持旧行为可去掉 clamp。

### 6.5 `_train_standard_step`（第 369-405 行）整体替换

```python
    def _train_standard_step(self, ldct, rdct, loss_meters, pbar, step, epoch):
        cfg = self.cfg
        model = self.model

        model_out = model(ldct)
        predict_for_viz = model_out

        if cfg.model.artifact_output and isinstance(model_out, tuple):
            # 伪影残差输出：(clean, artifact)；真实伪影 = 输入 - 干净GT
            pred_clean, pred_artifact = model_out
            gt_artifact = ldct - rdct
            loss, loss_dict = self._compute_loss(
                pred_clean, rdct, pred_artifact=pred_artifact, gt_artifact=gt_artifact)
            predict_for_viz = pred_clean

        elif cfg.loss.deep_supervision and isinstance(model_out, (tuple, list)):
            final_out = model_out[-1]
            loss, loss_dict = self._compute_loss(final_out, rdct)
            lc = cfg.loss
            for idx, out in enumerate(model_out[:-1]):
                target_resized = nn.functional.interpolate(
                    rdct, size=out.shape[2:], mode="bilinear", align_corners=False
                )
                if lc.mse_weight > 0:
                    loss += lc.mse_weight * self.criterion_mse(out, target_resized) * 0.1
            predict_for_viz = final_out

        else:
            loss, loss_dict = self._compute_loss(model_out, rdct)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_loss_meters(loss_meters, loss_dict, ldct.size(0))
        self.global_step += 1

        if step % cfg.training.log_interval == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        if step == 0 and (cfg.training.viz_interval > 0):
            self._log_images("train", ldct, rdct, predict_for_viz, epoch)
```

### 6.6 `_validate`（第 470-472 行）

```python
            predict = model(ldct)
            if self.cfg.model.artifact_output and isinstance(predict, (tuple, list)):
                predict = predict[0]      # artifact_output 时第一个输出是 clean
            elif isinstance(predict, (tuple, list)):
                predict = predict[-1]     # deep supervision 时最后一个输出
```

### 6.7 不改 GAN

`_train_gan_step` 假设 `model(ldct)` 返回单张图。**GAN 模式 + `artifact_output=true` 暂不支持**（adv_weight 保持 0 即可）。

---

## 7. 改动 6：新增配置文件 `experiments/denseunet_projct_TM.yaml`

```yaml
# DenseUNet2d — ProjCT + TMCode PBI 去伪影（伪影残差输出 + 复合损失）
experiment:
  name: DenseUNet_ProjCT_TM
  seed: 1234

data:
  reader: proj
  train_txt: ./train_proj_img.txt
  valid_txt: ./valid_proj_img.txt
  batch_size: 36
  valid_batch_size: 1
  num_workers: 8
  shuffle: true
  norm_clip_max: 0.28
  crop_rows: 16

model:
  name: DenseUNet2d
  in_chl: 1
  out_chl: 1
  model_chl: 32
  artifact_output: true            # 关键：伪影残差输出 + 恒等起点

loss:
  mse_weight: 0.0                  # 换用同学的复合损失
  adv_weight: 0.0                  # 关 GAN（默认 0.0，>0 开启）
  l1_weight: 0.3                   # L1
  artifact_weight: 0.1             # 伪影残差 L1
  ssim_weight: 0.8                 # 1 - SSIM
  fft_amp_weight: 0.8              # FFT 幅值
  angular_weight: 0.8              # 沿末维梯度
  sobel_weight: 0.3                # 多向 Sobel
  # charbonnier / vgg / tv / focal / ffl 保持 0

optimizer:
  name: Adam
  lr: 0.0001
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0

scheduler:
  name: StepLR
  step_size: 2
  gamma: 0.95

training:
  epochs: 200
  save_dir: ./runs/2026-0818-DenseUNet_ProjCT_TM
  log_dir: ./runs/2026-0818-DenseUNet_ProjCT_TM
  log_interval: 1
  viz_interval: 1
  save_interval: 20
```

训练命令：

```bash
python train.py --config experiments/denseunet_projct_TM.yaml
```

---

## 8. 验证方式

1. **冒烟测试**：先只跑 1 个 epoch（`--epochs 1`），确认不报错、损失分项（l1 / artifact / ssim / fft_amp / angular / sobel）都能打印出来。
   ```bash
   python train.py --config experiments/denseunet_projct_TM.yaml --epochs 1
   ```
2. **初始行为验证**：因为恒等起点，第 1 个 epoch 的 `artifact` 损失应接近 `L1(0, lq-gt)` 的初始值、`l1` 损失应接近输入与 GT 的初始 L1——说明模型确实从"恒等"开始。
3. **TensorBoard**：`train img/predict` 应显示干净的校正图（不是伪影图）。
4. **对照**：同数据下对比旧配置 `denseunet_projct.yaml` 与新配置的 valid MSE / SSIM / PSNR。

---

## 9. 常见坑

1. **`predict[-1]` 取错元素**：伪影输出是 `(clean, artifact)`，clean 在**第 0 个**。所有取值处（`_validate`、`_train_standard_step` 的 `_log_images`）都要按 `cfg.model.artifact_output` 分支，否则会把"伪影图"当结果画/算。
2. **artifact 不加 ReLU**：伪影可正可负，`conv6` 输出不要过 `F.relu`（同学原版也没加）。
3. **恒等初始化只对输出层**：只把 `self.end[0].conv` 近零化，不要动网络其它层。
4. **loss_meters 的 key 变了**：`_compute_loss` 现在会产出 `l1 / artifact / fft_amp / angular / sobel` 等分项，`_update_loss_meters` 会自动创建对应 meter，无需改。
5. **旧 checkpoint 兼容**：`artifact_output` 只在模型构造时生效，不改变权重结构；旧 checkpoint 仍可加载（等价于 artifact_output=false 的行为）。

---

## 10. 本次没做、但建议关注的两点

| 差异（见分析文档） | 本次状态 | 预期影响 / 建议 |
|---|---|---|
| 3.3 InstanceNorm vs BatchNorm | 不改，仍用 BN + batch 36 | 复合损失 + 伪影残差会带来主要提升；若发现仍不如同学，下一步把 BN 换成 InstanceNorm |
| 3.4 梯度裁剪 | 不改 | FFT / sobel / angular 损失梯度偏大，若训练震荡或发散，补一行即可：`torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)` |

---

## 11. 涉及文件清单

**新增：**
- `src/losses/sino.py`
- `experiments/denseunet_projct_TM.yaml`

**修改：**
- `src/losses/__init__.py` — 导出 3 个新损失
- `src/config.py` — LossConfig + ModelConfig 各加字段
- `src/models/unet.py` — 3 个模型加 `artifact_output` 参数，DenseUNet2d 实现伪影输出 + 恒等初始化
- `src/trainer.py` — `_build_model` / `_build_losses` / `_compute_loss` / `_train_standard_step` / `_validate`
