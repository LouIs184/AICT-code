# 投影域训练「简单像素讨好 / 关键区域装死」问题分析与解决方案

> 适用：投影域（ProjCT）训练，图像域（PhaseCT）的同类现象也可参考。
>
> 关联文档：[[PROJECTION_DOMAIN_CHANGES.md](./PROJECTION_DOMAIN_CHANGES.md)]、[[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)]、[[YAML_CROP_INTEGRATION.md](./YAML_CROP_INTEGRATION.md)]

---

## 一、问题现象

使用投影域训练（`experiments/denseunet_projct.yaml`，`DenseUNet2d` + 纯 MSE）时观察到：

- **整体残差（MSE / MAE）持续下降**，验证集整体 PSNR 缓慢上升；
- 但**难矫正的关键区域**（束硬化/散射伪影严重、高衰减结构——骨/金属——对应的投影通道）误差**不降反升**或长期停滞；
- 表现为模型"讨好"了大量简单像素（空气、均匀软组织的投影），对关键区域"装死"。

---

## 二、本质原因分析

### 2.1 为什么 MSE 会让模型"装死"（这不是 bug，是目标函数的必然结果）

MSE 的最优解是条件均值 $\mathbb{E}[y|x]$。当映射不适定、模型容量有限时，模型倾向输出"安全折中值"来压低平均损失。

关键的梯度直觉：

- 总损失是**逐像素平均平方误差**，而网络更新是**梯度平均**。一个裁剪后的探测器平面有 `112×512 ≈ 5.7 万` 个像素，关键区域可能只有几百到几千个。简单像素数量占绝对多数，且梯度方向高度一致，**硬像素的梯度被稀释**。
- 卷积核参数共享：修正一个硬像素会同时扰动其感受野内的所有像素。模型发现"保住简单像素"的期望收益 > "冒险修正硬像素"，于是收敛到把 `conv6` 输出压向 0、输出 ≈ 输入的**安全解**。
- `DenseUNet2d` 的输出是 `F.relu(x + conv6)`（见 `src/models/unet.py:166`），这是一个**残差结构**——模型默认"不做校正"，把输出抄回输入。这个结构设计进一步放大了"装死"倾向，需要损失函数去主动对抗它。

### 2.2 投影域的特殊性放大了问题

| 因素 | 说明 |
|---|---|
| 像素数量悬殊 | 空气、均匀软组织的投影占探测器平面绝大多数，对模型来说"天生简单" |
| 有效信号集中 | pbi 伪影的"需被校正"信号集中在高衰减结构的投影通道，区域极小 |
| 数值范围小 | 归一化后 `target - input` 差异可能只有 0.01~0.05（在 [0,1] 中），对 MSE 来说"不够醒目" |
| 逐视角 2D 建模 | 每个视角独立输入，而束硬化校正在 sinogram 上是**跨视角全局耦合**的。单视角 2D 模型缺乏上下文，可能根本"学不会"难区域——此时不是损失问题，是信息不足（见 6.3） |

### 2.3 核心结论

> 观察到的现象是 **MSE 目标函数的最优行为**。要解决，必须**主动改变优化目标**：让模型明确知道"关键区域的误差更贵"。

这必然以**牺牲部分整体 MSE** 为代价——你是在有意识地重新权衡目标，必须量化监控这个 trade-off（见第七章），否则会误判实验好坏。

---

## 三、解决方案总览

按"改动量小、见效快 → 改动大、收益上限高"排序。**建议先做 P0**。

| 优先级 | 方案 | 原理一句话 | 改动位置 |
|---|---|---|---|
| **P0** | 4.1 单调幂/对数变换 | 放大低值（高衰减）区的归一化差异，等价于给关键区加权重 | `loss` 内或 `proj_reader` |
| **P0** | 4.2 静态权重图 `w=|y−x|^α` | 校正量大的像素更值钱（域先验，最稳） | `src/losses/` 新增 |
| **P0** | 4.3 自适应焦点权重（focal） | 当前误差大的像素更值钱（自动适应，略不稳） | `src/losses/` 新增 |
| **P0** | 4.4 硬像素挖掘（top-K%） | 只回传最难 K% 像素的梯度，最激进 | `src/losses/` 新增 |
| **P0** | 4.5 幂损失 `mean(e^(2+γ))` | 大残差的影响力按指数放大 | `src/losses/` 新增 |
| **P1** | 5.1 逐视角 min-max 归一化 | 每个视角铺满 [0,1]，关键区动态范围变大 | `proj_reader.py` |
| **P1** | 5.2 硬视角采样 | 校正量大的视角多采 | `trainer.py` DataLoader |
| **P1** | 5.3 课程学习 | 先学整体，再逐步加大关键区权重 | 训练循环 |
| **P2** | 6.1 级联细化（两阶段） | 粗校正后专精残差大的区域 | `models/` |
| **P2** | 6.2 解码器空间注意力 | 让网络学会"往哪使劲" | `models/` |
| **P2** | 6.3 多视角上下文输入 | 给单视角 2D 模型补跨视角信息 | `proj_reader` + 模型 |
| **P2** | 6.4 图像域联合损失（可微 FBP） | 直接优化最终重建质量，治本 | 需引入重建算子 |
| **P3** | 7 关键区域分项指标 | 让"装死"可见，正确选 checkpoint | `trainer.py` / `test.py` |

---

## 四、P0：损失层改造（推荐先做，改动最小）

### 4.1 单调幂/对数变换（最便宜的"加权"）

原理：对 `pred` 和 `target` 同时施加单调变换 $f(x)=x^s\ (s<1)$ 或 $f(x)=\log(x+\epsilon)$。由于 $f'(x)=s x^{s-1}$ 在低值处更大，**低值（高衰减/关键）区域的同样绝对误差在变换空间里被放大**——等价于给关键区加了权重，且零参数、零显存开销。

```python
# 只改 loss 内比较对象，不改数据管线（因此推理端无需反变换）：
p = predict.clamp(min=0) ** 0.5
t = target.clamp(min=0) ** 0.5
loss = torch.mean((p - t) ** 2)
```

> 注意：这是在 loss 里做"加权"，**不需要**对数据集和推理输出做反变换，是最安全的接入方式。另一种做法是在 `proj_reader` 里对输入/目标都做变换（会改变输入分布，可能进一步帮助模型，但推理输出需做反变换）。

### 4.2 静态权重图 `w = |y − x|^α`（域先验，最稳，首选）

`|y − x|`（校正量，即 `proj_no_pbi − proj_pbi`）恰恰标出了"哪里需要被修正"。用它做逐像素权重，模型会优先处理伪影所在的探测器区域。

```python
def correction_magnitude_map(input_, target, power=1.0, wmax=20.0):
    """静态权重图：w = |target - input|^power，已归一化、封顶。"""
    w = (target - input_).abs().detach()          # 无梯度，域先验
    if power != 1.0:
        w = w ** power
    w = w / (w.mean() + 1e-8)                     # 归一化到均值=1，保持 loss 量级稳定
    w = torch.clamp(w, max=wmax)                  # 防止单像素主导
    return w

loss = (correction_magnitude_map(x, y) * (pred - y) ** 2).mean()
```

变体：
- 想要更偏重高衰减区，可改为 `w = (1 - target)^α`；
- 想抑制逐像素噪声，可先对 `w` 做高斯平滑（`src/utils/viz.py` 里已有 `gaussian_smooth` 可参考）；
- `α` 控制权重集中程度，建议从 `α=1.0` 起步，可试 `0.5` / `2.0`。

### 4.3 自适应焦点权重（focal）`w = |pred − y|^γ`

按**当前误差**动态加权：模型哪里错得多，哪里就获得更高权重。比静态权重更"主动"，但若 `γ` 过大容易震荡。

```python
err = (pred - target).abs().detach()              # 关键：detach，否则会放大梯度
w = (err + 1e-6) ** gamma                         # gamma 建议 0.5 ~ 2.0
w = w / (w.mean() + 1e-8)
w = torch.clamp(w, max=wmax)
loss = (w * (pred - target) ** 2).mean()
```

稳定性技巧：
- 用误差图的 **EMA**（滑动平均）代替单步误差，抑制帧间抖动；
- `γ` 从 `0.5` 起步，收敛稳定后再加大；
- 与普通 MSE 组合（见 8 章配置），别全替换。

### 4.4 硬像素挖掘（top-K%）

只回传**最难 K% 像素**的梯度，最直接、最激进。适合在 4.2/4.3 效果不够时使用。

```python
err = (pred - target).abs()
b, c, h, w = err.shape
flat = err.view(b, -1)
k = max(1, int(flat.shape[1] * keep_ratio))       # keep_ratio 建议 0.05 ~ 0.20
thr, _ = torch.topk(flat, k=k, dim=1, largest=True)
thr = thr[:, -1].view(b, 1, 1, 1)                 # 每样本第 k 大的误差作阈值
mask = (err >= thr).float()
loss = ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
```

### 4.5 幂损失 `mean(e^(2+γ))`

全像素都参与，但大残差按指数放大贡献。比 top-K 平滑，比纯 MSE 聚焦。

```python
e = (pred - target).abs()
loss = torch.mean(e ** (2 + gamma))   # gamma 建议 0.5 ~ 1.0
```

建议与少量普通 MSE 混合（如 `0.5 * MSE + 0.5 * 幂损失`），避免小误差区梯度消失导致卡住。

### 4.6 如何接入当前代码

#### 新增 `src/losses/hard.py`

把 4.2 / 4.3 / 4.4 / 4.5 封装成 `nn.Module`（`WeightedMSE` / `FocalMSE` / `TopKMSE` / `PowerLoss`），并在 `src/losses/__init__.py` 中导出。示例：

```python
"""src/losses/hard.py —— 关键区域聚焦损失。"""
import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    """MSE + 静态权重图 w=|target-input|^power。"""

    def __init__(self, power=1.0, wmax=20.0):
        super().__init__()
        self.power, self.wmax = power, wmax

    def forward(self, pred, target, input_):
        w = (target - input_).abs().detach() ** self.power
        w = w / (w.mean() + 1e-8)
        w = torch.clamp(w, max=self.wmax)
        return (w * (pred - target) ** 2).mean()


class FocalMSE(nn.Module):
    """MSE + 自适应焦点权重 w=|pred-target|^gamma。"""

    def __init__(self, gamma=1.0, wmax=20.0):
        super().__init__()
        self.gamma, self.wmax = gamma, wmax

    def forward(self, pred, target):
        w = (pred - target).abs().detach() + 1e-6
        w = w ** self.gamma
        w = w / (w.mean() + 1e-8)
        w = torch.clamp(w, max=self.wmax)
        return (w * (pred - target) ** 2).mean()


class TopKMSE(nn.Module):
    """只回传最难 keep_ratio% 像素的 MSE。"""

    def __init__(self, keep_ratio=0.1):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, pred, target):
        err = (pred - target).abs()
        b, _, h, w = err.shape
        flat = err.view(b, -1)
        k = max(1, int(flat.shape[1] * self.keep_ratio))
        thr, _ = torch.topk(flat, k=k, dim=1, largest=True)
        thr = thr[:, -1].view(b, 1, 1, 1)
        mask = (err >= thr).float()
        return ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
```

#### `src/config.py` 增加字段

```python
@dataclass
class LossConfig:
    ...
    hard_weight: float = 0.0        # >0 时启用关键区域损失
    hard_mode: str = "weighted"     # "weighted" | "focal" | "topk" | "power"
    hard_power: float = 1.0         # weighted 的指数
    hard_gamma: float = 1.0         # focal / power 的指数
    hard_keep_ratio: float = 0.1    # topk 保留比例
    hard_wmax: float = 20.0         # 权重上限（相对均值倍数）
    transform_gamma: float = 1.0    # <1 时对 pred/target 做幂变换（见 4.1）
```

#### `src/trainer.py::_compute_loss` 接入

```python
def _compute_loss(self, predict, target, model_out=None, input_tensor=None):
    lc = self.cfg.loss
    loss = 0.0
    losses = {}

    # 可选：幂变换（等价于暗区加权）
    p, t = predict, target
    if lc.transform_gamma != 1.0:
        p = p.clamp(min=0) ** lc.transform_gamma
        t = t.clamp(min=0) ** lc.transform_gamma

    if lc.mse_weight > 0:
        losses["mse"] = self.criterion_mse(p, t)
        loss += lc.mse_weight * losses["mse"]

    if lc.hard_weight > 0:
        if lc.hard_mode == "weighted":
            w = (t - input_tensor).abs().detach() ** lc.hard_power
            w = w / (w.mean() + 1e-8)
            w = torch.clamp(w, max=lc.hard_wmax)
            l_hard = (w * (p - t) ** 2).mean()
        elif lc.hard_mode == "focal":
            w = ((p - t).abs().detach() + 1e-6) ** lc.hard_gamma
            w = w / (w.mean() + 1e-8)
            w = torch.clamp(w, max=lc.hard_wmax)
            l_hard = (w * (p - t) ** 2).mean()
        elif lc.hard_mode == "topk":
            e = (p - t).abs()
            b, _, h, w_ = e.shape
            flat = e.view(b, -1)
            k = max(1, int(flat.shape[1] * lc.hard_keep_ratio))
            thr, _ = torch.topk(flat, k=k, dim=1, largest=True)
            thr = thr[:, -1].view(b, 1, 1, 1)
            mask = (e >= thr).float()
            l_hard = ((p - t) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        else:  # "power"
            l_hard = torch.mean((p - t).abs() ** (2 + lc.hard_gamma))
        losses["hard"] = l_hard
        loss += lc.hard_weight * l_hard
    ...
```

调用处：`_train_standard_step` / `_train_gan_step` / 深度监督分支把 `input_tensor=ldct` 传入（`_train_standard_step` 中即 `loss, loss_dict = self._compute_loss(final_out, rdct, input_tensor=ldct)`）。

---

## 五、P1：数据层改造

### 5.1 逐视角 min-max 归一化

当前 `proj_reader` 用固定 `clip(x, 0, 0.28)/0.28`。改为**每个视角独立 min-max**，让每个样本铺满 [0,1]，关键区的归一化差异变大：

```python
# src/data/proj_reader.py 中：
v_min, v_max = input_view.min(), input_view.max()
input_view = (input_view - v_min) / (v_max - v_min + 1e-8)
# target 用同一组参数归一化
label_view = (label_view - v_min) / (v_max - v_min + 1e-8)
```

- 由于每个样本已经是单视角，min-max 是逐样本的，训练/推理天然一致；
- 注意与 `crop_rows` 的先后：先 crop 再 min-max（黑边裁剪后再统计，数值才稳定）。

### 5.2 硬视角采样

按校正量给视角加权采样，让"含伪影信息多"的视角更多被看到：

```python
# 预计算每个视角的权重（train 集合）
def view_score(view_input, view_target):
    return (view_target - view_input).abs().mean().item()

weights = [view_score(x, y) for (x, y) in train_views]   # 或 |·|^α 更激进
sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
train_loader = DataLoader(..., sampler=sampler)
```

### 5.3 课程学习

先跑普通 MSE（或小权重）若干 epoch 让模型学会整体结构，再逐步加大 `hard_weight` / `γ` / 缩小 `keep_ratio`。避免一开始就逼模型死磕硬像素导致震荡。可以在 trainer 里按 epoch 线性调度：

```python
hard_w = base_hard * min(1.0, epoch / warmup_epochs)
```

---

## 六、P2：模型 / 架构层

### 6.1 级联细化（两阶段）

第一遍输出粗校正 `ŷ₁`，算出误差图，第二遍（浅层子网）只对**残差大的区域**做增量修正：

```python
y1 = model(x)
err = (y1 - target).abs().detach()
w = err / (err.mean() + 1e-8)          # 或 (err > quantile).float()
delta = refine(y1)                     # 小网络，输入含 y1 和 w
y2 = y1 + delta * w                    # 只在大误差区域显著修改
loss = MSE(y1, target) * 0.5 + MSE(y2, target) * 0.5
```

### 6.2 解码器空间注意力

让网络学会"往哪使劲"：在解码器每层把上一层的输出经过一个 `1×1` 卷积 + `sigmoid` 生成空间注意力图，对特征加权。DenseUNet 的 `StackResDecoder2d`（`src/models/blocks.py`）里加注意力即可，改动局部。

### 6.3 多视角上下文输入（in_chl = K）

束硬化校正在 sinogram 上是全局的，**单视角 2D 模型可能缺少跨视角信息**。若做了 P0/P1 后关键区仍然毫无起色，优先怀疑这一条——把相邻 `K` 个视角堆叠作为输入（`in_chl=K`，如 3 或 5），让模型看到局部 sinogram 上下文：

```python
# proj_reader: 返回连续 K 个视角 (K, H, W) 作为输入，目标仍是中心视角
K = 3
input_view = volume[view_idx - K//2 : view_idx + K//2 + 1]   # 边缘视角 pad 或跳过
```

这是结构性改动，但直指投影域"难区域学不会"的信息不足根源。

### 6.4 图像域联合损失（可微 FBP）—— 最贴近最终目标

最终质量在图像域衡量。如果让投影域损失 + **可微重建**（如 `torch-fbp`、`astra-toolbox`）构成的图像域损失联合监督，模型直接为"重建图像对、关键部位对"负责，是从根上解决问题。但注意：

- 需要知道准确的扫描几何（720 视角 × 128 行 × 512 通道是扇形束还是锥形束、探测器间距等），几何错误会让图像域损失失真甚至有害；
- 当前仓库没有重建算法，需要新增依赖；建议先做 P0/P1，把它作为后续方案；
- 若加上，图像域损失只需监督**少数几个重建切片**即可，不必每步都重建整个体数据。

---

## 七、P3：监控与评估（让"装死"可见，正确选 checkpoint）

**这是开始任何实验前必须先做的**。否则整体指标上升掩盖关键区退化，你会被误导。

### 7.1 关键区域分项指标

在 `_validate`（`src/trainer.py:409`）里按**校正量**定义关键区（域先验，不依赖模型），分别统计：

```python
corr = (rdct - ldct).abs()                                  # 校正量
thr = torch.quantile(corr.view(corr.size(0), -1), 0.9, dim=1)
thr = thr.view(-1, 1, 1, 1)
hard_mask = (corr >= thr).float()
easy_mask = 1.0 - hard_mask

err2 = (predict - rdct) ** 2
hard_mse = (err2 * hard_mask).sum() / (hard_mask.sum() + 1e-8)
easy_mse = (err2 * easy_mask).sum() / (easy_mask.sum() + 1e-8)

# writer 里记录：
#   valid/hard_psnr = -10*log10(hard_mse)
#   valid/easy_psnr = -10*log10(easy_mse)
#   valid/hard_easy_ratio = hard_mse / easy_mse
```

`hard_easy_ratio` 若随 epoch **上升**，说明模型正在对关键区"装死"——即使整体 MSE 在下降。

### 7.2 残差热图

`_log_images`（`src/trainer.py:449`）已经记录了整图 residual，额外加一张**只保留关键区的残差图**，肉眼确认关键区是否在改善。

### 7.3 图像域最终评估

投影域指标**不能**解释为 HU。最终验收应把 `pred.raw` 用正确几何重建到图像域，再算 HU 域的 PSNR/SSIM。参见 [[PROJECTION_DOMAIN_CHANGES.md](./PROJECTION_DOMAIN_CHANGES.md)] 第十节。在引入重建前，用 7.1 的分项指标作为代理。

---

## 八、推荐实验矩阵

> 原则：**每次只改一个变量**，全部记录 hard/easy 分项指标。

| 序号 | 配置 | 目的 | 判断标准 |
|---|---|---|---|
| 0 | 当前基线（纯 MSE，`denseunet_projct.yaml`） | 复现问题，打底 | 记录 hard/easy 分项，确认"装死" |
| 1 | `mse_weight:0.5` + `hard_weight:1.0` + `hard_mode:weighted` | 校正量大的区域优先 | hard_psnr 上升，整体略降可接受 |
| 2 | 同 1，但 `hard_mode:focal, hard_gamma:0.5` | 自适应 | 与 1 对比稳定性与 hard 提升 |
| 3 | 同 1，加 `transform_gamma:0.5` | 叠加幂变换 | 低值区动态范围放大 |
| 4 | `hard_mode:topk, hard_keep_ratio:0.1` | 最激进 | hard 显著改善但整体 MSE 掉得多，观察值不值 |
| 5 | 1 + 硬视角采样 | 数据层叠加 | 应稳定优于 1 |
| 6 | 多视角输入 `in_chl=3`（若 1~5 后 hard 仍不动） | 补跨视角信息 | 结构性验证 |

参数起点建议：

| 参数 | 起点 | 范围 |
|---|---|---|
| `hard_weight` | 1.0 | 0.2 ~ 5.0 |
| `hard_power`（weighted） | 1.0 | 0.5 ~ 2.0 |
| `hard_gamma`（focal） | 0.5 | 0.25 ~ 2.0 |
| `hard_keep_ratio`（topk） | 0.1 | 0.05 ~ 0.20 |
| `hard_wmax` | 20 | 10 ~ 50 |
| `transform_gamma` | 0.5 | 0.25 ~ 1.0 |

---

## 九、注意事项与常见坑

1. **权重图必须 `detach()`**：`w` 不能参与梯度回传，否则等于把损失变成更高次幂并放大梯度，易爆。
2. **权重要归一化到均值 1 + 上限裁剪**：否则 loss 量级漂移，需要重调 lr / 损失权重。
3. **保留少量普通 MSE 保底**：全换成 hard 损失会让简单区梯度消失、模型震荡或过拟合到噪声。用 `mse_weight:0.5` 这类"保底"组合。
4. **focal / 幂损失会放大噪声**：投影数据在高衰减区 SNR 低，过大 `γ` 会把噪声当"难像素"死磕。配合 `hard_wmax`、EMA 误差图、课程学习缓解。
5. **变换一致性**：4.1 的幂变换若只在 loss 内做，推理端无需反变换；若在 `proj_reader` 里对输入/目标都做，则**推理输出必须反变换**，且 min-max/变换参数要与训练一致。
6. **不要只盯整体 MSE 选 checkpoint / 早停**：用 7.1 的 hard 分项，或最终图像域指标。
7. **crop_rows 与 min-max 顺序**：先 crop 后归一化（见 5.1）。
8. **这是目标权衡，不是免费午餐**：提高关键区质量会牺牲部分整体 MSE。实验中明确记录"牺牲多少整体、换来多少关键区"，由任务需求决定取舍。
9. **先确认是不是损失问题**：如果 P0+P1 全部做完关键区仍纹丝不动，大概率是**信息不足**（6.3 多视角 / 6.4 图像域损失）而非权重问题——单视角 2D 模型可能压根看不见跨视角的束硬化耦合信息。

---

## 十、参考资料

- 投影域改造背景：[[PROJECTION_DOMAIN_CHANGES.md](./PROJECTION_DOMAIN_CHANGES.md)]
- 训练/推理/配置使用：[[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)]
- 投影域黑边裁剪：[[YAML_CROP_INTEGRATION.md](./YAML_CROP_INTEGRATION.md)]
- Focal loss 思想：Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
- 可微重建：`torch-fbp`、`astra-toolbox`
- 不确定性加权（注意其"装死"风险，见 2.1）：Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
