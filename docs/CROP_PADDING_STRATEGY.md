# 投影域黑边裁剪与 U-Net 下采样对齐策略

> 问题：投影数据 (720, 128, 512) 每个视角的 128 行上下存在黑边（无效行），裁剪后会破坏 2 的幂次对齐，导致 U-Net 下采样失败。本文档分析所有可行的解决方案。

---

## 一、问题定义

### 现状

| 项目 | 值 |
|------|-----|
| 探测器原始行数 | 128 |
| 探测器通道数 | 512 |
| 黑边位置 | 最上面约 10 行 + 最下面约 10 行（无效数据） |
| 有效区域 | 中间约 108 行 |
| 裁剪目标 | `(128, 512)` → `(108, 512)` |

### 为什么 108 有问题

当前 DenseUNet2d 包含 **4 次 stride=2 下采样**（使用 `nn.MaxPool2d(kernel_size=2, stride=2)` 或 `nn.Conv2d(stride=2)`）：

```
输入 (1, H, W)
  ↓ down1  stride=2   →  H/2
  ↓ down2  stride=2   →  H/4
  ↓ down3  stride=2   →  H/8
  ↓ down4  stride=2   →  H/16
  ↓ up4    stride=2   →  2×H/16  (上采样恢复)
  ...
```

**要求**：`H` 必须能被 `2^4 = 16` 整除，且**每一层的中间特征图分辨率都必须是整数**。

| 高度 H | 下采样 1 次 | 下采样 2 次 | 下采样 3 次 | 下采样 4 次 | 是否合法 |
|--------|------------|------------|------------|------------|---------|
| 128 | 64 | 32 | 16 | 8 | ✅ |
| **108** | **54** | **27** | **13.5** ❌ | **6.75** ❌ | ❌ 奇数，无法整除 |
| 112 | 56 | 28 | 14 | 7 | ✅ |
| 96 | 48 | 24 | 12 | 6 | ✅ |
| 80 | 40 | 20 | 10 | 5 | ✅ |

---

## 二、可选的解决方案（共 5 种）

---

### 方案 A：调整裁剪量，对齐到 16 的倍数（推荐）

不要裁剪到 108，而是**调整上下裁剪的行数，使最终高度为 16 的倍数**。

#### 选项 A1：裁剪到 112（上 8 + 下 8）

```
原始:    128
裁剪:   上 8 + 下 8 = 16
结果:    112  ← 112 ÷ 16 = 7，完美对齐
```

**优点**：
- ✅ 无需改动 U-Net 任何代码
- ✅ 112 是 16 的倍数，各层下采样结果均为整数：112→56→28→14→7
- ✅ 只多保留了几行，基本不影响去黑边的目的
- ✅ 512 维度本身是 16 的倍数（512÷16=32），无问题

**缺点**：
- ⚠️ 上 8 + 下 8 不一定刚好覆盖全部黑边——如果黑边正好是 10 行，会残留 2 行半黑边
- ⚠️ 但残留 2 行黑边的影响远小于保留全部 20 行黑边

#### 选项 A2：裁剪到 96（上 16 + 下 16）

```
原始:    128
裁剪:   上 16 + 下 16 = 32
结果:    96   ← 96 ÷ 16 = 6，对齐
```

**优点**：
- ✅ 无需改动 U-Net
- ✅ 96 是 16 的倍数
- ✅ 彻底去除黑边（上 16、下 16 一定超过黑边范围）

**缺点**：
- ⚠️ 可能裁剪掉部分有效信号（如果有效区域真的到了第 10~118 行，上 16 下 16 会切掉一些边缘组织）

---

### 方案 B：Padding 到 16 的倍数（零填充或反射填充）

裁剪到 108，再 padding 到 112（或其他 16 的倍数）。

```
裁剪:   (128, 512) → (108, 512)  ← 去掉上下黑边
Padding: (108, 512) → (112, 512)  ← pad=2（上下各 2 行）
        或者       → (128, 512)  ← pad=10（上下各 10 行）
```

**在 DataLoader/预处理中实现（推荐方式）**：

```python
import torch.nn.functional as F

def crop_and_pad(proj_view, crop_top=10, crop_bottom=10, target_rows=112):
    """
    proj_view: shape (H, W) 或 (1, H, W)
    裁剪黑边后 padding 到 target_rows 对齐 16
    """
    # 步骤 1：裁剪黑边
    cropped = proj_view[..., crop_top:-crop_bottom, :]  # (108, 512)
    
    # 步骤 2：Padding 到 target_rows
    pad_top = 0
    pad_bottom = target_rows - cropped.shape[-2]  # 112 - 108 = 4
    # F.pad: (left, right, top, bottom)
    padded = F.pad(cropped, (0, 0, pad_top, pad_bottom), mode='replicate')
    # 用 replicate 或 reflect 模式比 zero padding 更好（边界平滑过渡）
    return padded
```

**padding mode 选择**：

| 模式 | 行为 | 推荐度 |
|------|------|--------|
| `constant` (默认=0) | 填 0 | ❌ 引入新黑边，破坏裁剪目的 |
| `replicate` | 复制边界值填充 | ✅ 推荐，边界平滑 |
| `reflect` | 镜像反射 | ✅ 也不错，但边界可能有轻微不连续 |

**优点**：
- ✅ 保留精确的 108 行有效区域（想留多少留多少）
- ✅ 无需改网络结构

**缺点**：
- ⚠️ padding 的行虽然用 replicate 比零填充好，但仍然是**人工引入的数据**
- ⚠️ 训练时 padding 区域算 loss 吗？如果不算，需要 mask；如果算，会引入 bias
- ⚠️ 实现了"裁剪黑边→重新造边"，逻辑上不够干净

#### 关于 loss 计算的重要性说明

如果使用 padding，**必须确保 loss 只计算有效区域（原始 108 行），不计算 padding 部分**：

```python
class MaskedLoss(nn.Module):
    """只计算有效区域的 loss，忽略 padding 区域"""
    def __init__(self, base_loss=nn.MSELoss(), valid_rows=108, total_rows=112):
        super().__init__()
        self.base_loss = base_loss
        self.valid_rows = valid_rows
        self.total_rows = total_rows
    
    def forward(self, pred, target):
        # pred, target: (B, 1, total_rows, 512)
        # 只取前 valid_rows（row 方向裁剪掉 padding 行）
        pred_valid = pred[..., :self.valid_rows, :]
        target_valid = target[..., :self.valid_rows, :]
        return self.base_loss(pred_valid, target_valid)
```

---

### 方案 C：修改 U-Net，接受任意尺寸（使用 Adaptive Pooling / padding）

改动 DenseUNet2d，让下采样模块遇到奇数尺寸时自动 padding：

```python
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        # standard padding=1 gives same padding only for even sizes
    
    def forward(self, x):
        # 如果 H 或 W 是奇数，自动 pad 成偶数
        if x.shape[-2] % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='replicate')
        return self.conv(x)
```

**或者更彻底的方案**：在输入层检测尺寸并自动 padding 到 16 的倍数：

```python
class AutoPadUNet(DenseUNet2d):
    """在入口自动 pad 到 16 的倍数"""
    def forward(self, x):
        # 记录原始尺寸
        orig_h, orig_w = x.shape[-2:]
        
        # 计算需要 pad 到 16 的倍数
        pad_h = (16 - orig_h % 16) % 16
        pad_w = (16 - orig_w % 16) % 16
        
        # padding: (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        
        # U-Net forward
        x = super().forward(x)
        
        # 裁剪回原始尺寸
        return x[..., :orig_h, :orig_w]
```

**优点**：
- ✅ 输入尺寸完全灵活，想裁多少裁多少
- ✅ 裁剪 + padding 逻辑封装在模型内部，DataLoader 不需要操心

**缺点**：
- ⚠️ 需要改模型代码，如果后续要加载预训练权重可能不兼容
- ⚠️ 由于 `CeLU` 等激活函数的存在，padding 区域的响应可能在上采样路径中影响有效区域（通过 skip connection）

---

### 方案 D：改变下采样次数

将 DenseUNet2d 的最后一次下采样去掉，只下采样 3 次：

```
修改后下采样路径：
输入 (1, 108, 512)
  ↓ down1  stride=2   →  54
  ↓ down2  stride=2   →  27
  ↓ down3  stride=2   →  13  ← 最后一次下采样后 27→13（int，用 ceil 模式）
  ↓ center
  ↓ up3    stride=2   →  26 → 上采样后 13→26
  ↓ up2    stride=2   →  52
  ↓ up1    stride=2   →  104 → 最终输出 104 ≠ 108 ❌
```

**问题**：下采样 3 次后上采样 3 次，输入 108 → 输出 104（因为奇数下采样用 ceil 会吞掉 1 个像素）。除非同时修改上采样层的输出通道数对齐方式，否则输出和输入尺寸不匹配。

**结论**：此方案改动大、容易出 bug，**不推荐**。

---

### 方案 E：调整 U-Net 的输入层，用 ConvTranspose 升维

在 U-Net 入口加一个转置卷积层，将 `(108, 512)` 升采样到 `(128, 512)`：

```python
class ResizeInputUNet(DenseUNet2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用转置卷积将 108→128（或直接用插值）
        self.resize = nn.Upsample(size=(128, 512), mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.resize(x)       # (108, 512) → (128, 512)
        x = super().forward(x)   # 标准 U-Net
        x = x[..., :108, :]      # 裁回原始尺寸（或用插值缩回）
        return x
```

**优点**：
- ✅ U-Net 主体完全不动
- ✅ 108→128 是升采样，信息无损

**缺点**：
- ⚠️ 插值会轻微模糊有效区域的边界
- ⚠️ 增加了计算量（108→128 增加了约 18% 的像素）

---

## 三、方案对比汇总

| 方案 | 需改代码 | 数据损失 | 效果 | 推荐度 |
|------|---------|---------|------|--------|
| **A1: 裁到 112** | 仅裁剪行数参数 | 残留 2 行黑边 | 黑边减少 90%，U-Net 无需改动 | ⭐⭐⭐⭐⭐ |
| **A2: 裁到 96** | 仅裁剪行数参数 | 可能多裁 | 彻底去黑边，但可能损失边缘信号 | ⭐⭐⭐⭐ |
| **B: 裁到 108 + pad** | DataLoader + loss mask | None | 保留 108 有效行，但需 loss mask | ⭐⭐⭐⭐ |
| **C: 改 U-Net 自动 pad** | 模型代码 | None | 最灵活，但耦合度高 | ⭐⭐⭐ |
| **D: 改下采样次数** | 模型代码 | None | 方案复杂，奇偶问题处理麻烦 | ⭐ |
| **E: 入口升采样** | 模型代码 | 轻微模糊 | 保留 108 有效行，额外计算 | ⭐⭐⭐ |

---

## 四、推荐方案（详细）

### 首选：方案 A1 — 裁剪到 112（上下各 8）

**为什么选这个**：
1. 改动最小 —— 只需要改 `prepare_proj_data.py` 或 `read_proj_file.py` 中的裁剪行数
2. 112 是 16 的倍数，U-Net 完全兼容
3. 上下各去 8 行（共 16 行）已经去掉了绝大部分黑边区域
4. 不需要 loss mask、不需要改模型、不需要额外的 forward hook

**具体操作**：

```python
# 在 DataLoader 中（修改 datasets/read_proj_file.py）

import numpy as np
import torch

class proj_reader(Dataset):
    def __init__(self, paired_img_txt, crop_top=8, crop_bottom=8, ...):
        # 读取索引文件
        ...
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
    
    def __getitem__(self, index):
        # 读取完整投影数据: (720, 128, 512)
        proj = np.fromfile(self.proj_path, dtype=np.float32).reshape(720, 128, 512)
        
        # 取指定视角: (128, 512)
        view = proj[self.view_idx]
        
        # 上下裁剪黑边: (128, 512) → (112, 512)
        if self.crop_bottom > 0:
            view = view[self.crop_top:-self.crop_bottom, :]
        else:
            view = view[self.crop_top:, :]
        
        # 归一化, 加通道维度, 返回
        ...
```

**做了下采样验证**：

```
输入 (1, 112, 512)
  ↓ down1   → (32, 56, 256)    ✅ 112/2=56
  ↓ down2   → (64, 28, 128)    ✅ 56/2=28
  ↓ down3   → (128, 14, 64)    ✅ 28/2=14
  ↓ down4   → (256, 7, 32)     ✅ 14/2=7
  ↓ center  → (512, 7, 32)     ✅
  ↓ up4     → (256, 14, 64)    ✅ 7*2=14
  ↓ up3     → (128, 28, 128)   ✅
  ↓ up2     → (64, 56, 256)    ✅
  ↓ up1     → (32, 112, 512)   ✅ 恢复原始尺寸
  ↓ end     → (1, 112, 512)    ✅
```

所有中间特征图分辨率均为整数，没问题。

---

### 备选：方案 B — 裁 108 + padding（如果不能接受残留黑边）

如果：
- 确认黑边恰好在第 10~117 行之间，上下各 10 行
- 裁剪 8 行后仍能看到残留半黑边
- 必须保留不超过 108 行有效区域

那就用方案 B。

**DataLoader 实现**：

```python
# datasets/read_proj_file.py

class proj_reader(Dataset):
    def __init__(self, paired_img_txt, crop_top=10, crop_bottom=10, pad_to=112):
        ...
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.pad_to = pad_to  # 必须是 16 的倍数
    
    def __getitem__(self, index):
        # ...读取视角...
        view = proj[self.view_idx]  # (128, 512)
        
        # 裁剪黑边
        cropped = view[self.crop_top:-self.crop_bottom, :]  # (108, 512)
        
        # Padding 到 112
        pad_h = self.pad_to - cropped.shape[0]  # 112 - 108 = 4
        # replicate padding（复制边界值），只 padding 底部
        padded = np.pad(cropped, 
                        ((0, pad_h), (0, 0)), 
                        mode='edge')  # edge = replicate
        
        return padded
```

**Loss mask**（可选，但推荐）：

如果 padding 的 4 行参与 loss 计算，会让模型学习在底部产生有效值（实际上底部是 pad 出来的），这不好。所以推荐在 loss 上加 mask：

```python
# train_unet_proj_2d.py 中的 loss 计算

def loss_fn(pred, target, valid_rows=108):
    """只计算前 valid_rows 行的 loss"""
    pred_valid = pred[..., :valid_rows, :]    # (B, 1, 108, 512)
    target_valid = target[..., :valid_rows, :]  # (B, 1, 108, 512)
    return F.mse_loss(pred_valid, target_valid) + charbonnier_loss(pred_valid, target_valid)
```

---

## 五、验证：原始 U-Net 的输入尺寸兼容性

当前 `DenseUNet2d` 的具体结构（基于 `PROJECTION_DOMAIN_CHANGES.md` 和 `TRAINING_GUIDE.md` 的分析）：

### 下采样路径

| 名称 | 操作 | 通道变化 | H×W（128） | H×W（112） | H×W（108→112pad） |
|------|------|---------|-----------|-----------|------------------|
| begin | Conv 3×3 | 1→32 | 128×512 | 112×512 | 112×512 |
| down1 | Pool/2 | 32→32 | 64×256 | 56×256 | 56×256 |
| down2 | Conv/DS | 32→64 | 32×128 | 28×128 | 28×128 |
| down3 | Conv/DS | 64→128 | 16×64 | 14×64 | 14×64 |
| down4 | Conv/DS | 128→256 | 8×32 | 7×32 | 7×32 |
| center | Dense | 256→512 | 8×32 | 7×32 | 7×32 |

### 上采样路径

| 名称 | 操作 | H×W（128） | H×W（112） |
|------|------|-----------|-----------|
| up4 | Up/2 + Conv | 16×64 | 14×64 |
| up3 | Up/2 + Conv | 32×128 | 28×128 |
| up2 | Up/2 + Conv | 64×256 | 56×256 |
| up1 | Up/2 + Conv | 128×512 | 112×512 |
| end | Conv 1×1 | 1×512 | 1×512 |

**验证结论**：

| 输入高度 | 验证结果 | 说明 |
|---------|---------|------|
| 128 (原始) | ✅ | 全 16 整除 |
| **112** | **✅** | **全 16 整除（112/16=7）** |
| 96 | ✅ | 全 16 整除（96/16=6） |
| 80 | ✅ | 全 16 整除（80/16=5） |
| 108 | ❌ | 下采样 2 次后 27→奇数，无法整除 |
| 110 | ❌ | 110/2=55，奇数 |
| 100 | ❌ | 100/4=25，奇数 |

### 512 通道维度

512 = 2^9，是 16 的倍数，无论哪种方案都不需要处理。

---

## 六、关于 CeLU 和奇数尺寸的说明

UU-Net 中使用 `CeLU` 激活函数。CeLU 本身不改变张量形状，所以只要**分辨率（H 和 W）在各层都是整数**，CeLU 不会引入任何问题。

问题完全出在 **MaxPool2d(stride=2) 或 Conv2d(stride=2)** 上：
- 当输入尺寸为偶数（如 28）：28/2=14，完美。
- 当输入尺寸为奇数（如 27）：27/2=13.5，`ceil_mode=True` 会取 14，`ceil_mode=False` 会取 13。
  - `ceil_mode=True`: 13.5 → 14（向下取整的 ceil），但上采样时 14×2=28 ≠ 27，**形状不匹配**。
  - `ceil_mode=False`: 13.5 → 13，上采样时 13×2=26 ≠ 27，**同样不匹配**。

所以结论是 **U-Net 要求输入尺寸能被 2^(下采样次数) 整除是硬约束**，不能绕过。

---

## 七、实际数据建议

### 建议的实施步骤

1. **先可视化确认黑边的精确位置**：
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   
   proj = np.fromfile("proj_no_pbi.raw", dtype=np.float32).reshape(720, 128, 512)
   
   # 看第 360 个视角（中间视角）
   view = proj[360]
   
   # 看每行的平均强度
   row_means = view.mean(axis=1)  # 每行的平均像素值
   
   # 找非零（有效）行的起始和结束
   nonzero_rows = np.where(row_means > 1e-6)[0]
   print(f"有效行范围: {nonzero_rows[0]} ~ {nonzero_rows[-1]}")
   print(f"有效行数: {len(nonzero_rows)}")
   
   # 可视化
   plt.figure(figsize=(12, 4))
   plt.subplot(121)
   plt.imshow(view, aspect='auto', cmap='gray')
   plt.subplot(122)
   plt.plot(row_means)
   plt.axvline(x=nonzero_rows[0], color='r', linestyle='--')
   plt.axvline(x=nonzero_rows[-1], color='r', linestyle='--')
   plt.show()
   ```

2. **根据实际黑边宽度选择方案**：
   - 如果黑边每侧 ≤ 8 行 → 用方案 A1（裁剪到 112，上下各 8）
   - 如果黑边每侧 ≤ 12 行 → 用方案 B（裁剪到 108 + pad 到 112）
   - 黑边范围不固定 → 用方案 C（自动 padding 的 U-Net）

3. **推荐的选择矩阵**：

   | 黑边情况 | 推荐方案 | 裁剪尺寸 | 最终送入 U-Net 的尺寸 |
   |---------|---------|---------|-------------------|
   | 上下各 8 行黑边 | A1 | 112 | 112（原生对齐） |
   | 上下各 10 行黑边 | B | 108 | 112（pad 4 行） |
   | 上下各 12 行黑边 | B | 104 | 112（pad 8 行） |
   | 上下各 15 行黑边 | A2 | 98→96 (对齐) | 96（原生对齐） |
   | 范围每视角不同 | C | 动态 | 自动 pad 到 16 倍数 |

---

## 八、总结

| 问题 | 答案 |
|------|------|
| 108 不能直接送入 U-Net？ | ❌ 不行，第二次下采样会出奇数 27 |
| 可以 padding 吗？ | ✅ 可以，但推荐用 `replicate` 模式，且 loss 需要 mask |
| 最简单的方式？ | ✅ **方案 A1：裁到 112**（上下各 8 行），零代码改动 |
| 如果必须裁到 108？ | ✅ 方案 B：裁到 108 → replicate pad 到 112 → loss mask |
| 对模型精度有影响吗？ | 方案 A1/B 几乎无影响，因为黑边区域本身无有效信号 |

**最终推荐**：先确认黑边行数。如果上下各 ≤ 8 行，用方案 A1（裁剪到 112），这是最优解。
