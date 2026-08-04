# 自适应焦点权重 FocalMSE（4.3）接入与使用说明

> 对应 [[HARD_REGION_OPTIMIZATION_GUIDE.md](./HARD_REGION_OPTIMIZATION_GUIDE.md)] 4.3 节的实现。
> 目的：给"当前误差大的像素"更高权重，对抗纯 MSE 下模型对关键区域（束硬化/散射伪影）"装死"的倾向。
> 注意：focal 是**目标权衡**，会牺牲部分整体 MSE 来换取关键区质量，请量化监控（见原文档第七章）。

---

## 一、改动了哪些文件

共 4 个文件，全部为**增量改动**，未改任何原有调用逻辑：

| 文件 | 改动 |
|---|---|
| `src/losses/hard.py` | **新增**。定义 `FocalMSE`（核心），按 4.3 实现：`w = \|pred − target\|^gamma`，含 EMA 平滑选项 |
| `src/losses/__init__.py` | 导出 `FocalMSE`，加入 `__all__` |
| `src/config.py` | `LossConfig` 新增 4 个字段：`hard_weight` / `hard_gamma` / `hard_wmax` / `hard_ema_decay` |
| `src/trainer.py` | ① `import` 引入 `FocalMSE`；② `_build_losses()`：`hard_weight>0` 时实例化 `self.criterion_focal`；③ `_compute_loss()`：`hard_weight>0` 时把 focal 项加入总损失 |

调用点（`_train_standard_step` / `_train_gan_step` / 深度监督）**无需改动**——focal 只用 `pred`/`target`，不依赖网络输入。

### `FocalMSE` 实现要点（`src/losses/hard.py`）

```python
class FocalMSE(nn.Module):
    def __init__(self, gamma=1.0, wmax=20.0, ema_decay=0.0):
        ...
    def forward(self, pred, target):
        err = (pred - target).abs().detach()   # 关键：detach，权重不参与梯度
        # 可选：误差图 EMA 平滑，抑制帧间抖动
        if self.ema_decay > 0:
            ...  # ema_err = ema_decay*ema_err + (1-ema_decay)*err
            err = self.ema_err
        w = (err + 1e-6) ** self.gamma
        w = w / (w.mean() + 1e-8)              # 归一化到均值=1，loss 量级稳定
        w = torch.clamp(w, max=self.wmax)      # 防止单像素主导
        return (w * (pred - target) ** 2).mean()
```

- 权重图已 `detach()`，不会放大梯度。
- EMA 用普通属性而非 `buffer`，不会进 checkpoint；训练中若 batch 尺寸变化会自动重置。

---

## 二、如何使用

在实验 YAML（如 `experiments/denseunet_projct.yaml`）的 `loss:` 段加入 focal 字段即可，**不开则行为与原来完全一致**：

```yaml
loss:
  mse_weight: 0.5          # 保底 MSE（建议保留，别全替换成 focal）
  hard_weight: 1.0         # focal 权重，>0 时启用；起点建议 0.5 ~ 1.0，范围 0.2 ~ 5.0
  hard_gamma: 0.5          # focal 指数；起点 0.5（收敛稳定后再加大），范围 0.25 ~ 2.0
  hard_wmax: 20.0          # 权重上限（相对均值倍数），范围 10 ~ 50
  hard_ema_decay: 0.0      # 误差图 EMA 平滑，0=不用；若训练震荡可试 0.9
```

训练时总损失 = `mse_weight * MSE + hard_weight * FocalMSE(+ 其他已配置项)`。TensorBoard 中会多出 `loss/focal` 分项。

**推荐实验顺序**（参照原文档第八章）：
1. 基线：`mse_weight: 1.0`，不开 hard —— 复现"装死"，记录关键区分项指标；
2. `mse_weight: 0.5` + `hard_weight: 1.0` + `hard_gamma: 0.5` —— 对比 focal 是否提升关键区、稳定如何；
3. 稳定后再逐步加大 `hard_gamma` / `hard_weight`。

---

## 三、注意事项（摘自原文档第九章）

1. **保留少量普通 MSE 保底**：全用 focal 会让简单区梯度减弱、模型震荡或过拟合噪声。`mse_weight: 0.5` 这类组合是推荐的。
2. **focal 会放大噪声**：投影数据在高衰减区 SNR 低，过大的 `hard_gamma` 会把噪声当"难像素"死磕。配合 `hard_wmax` 上限、EMA 平滑（`hard_ema_decay: 0.9`）缓解。
3. **γ 从 0.5 起步**，收敛稳定后再加大，避免一开始就震荡。
4. **不要只盯整体 MSE 选 checkpoint**：focal 本质上牺牲整体换关键区，请用关键区（hard/easy）分项指标或最终图像域指标判断（原文档 7.1）。
5. 若 P0 全部做完关键区仍纹丝不动，大概率是**信息不足**（原文档 6.3 多视角输入 / 6.4 图像域损失）而非权重问题。

---

## 四、验证

跑通后确认两件事：
- 训练日志 / TensorBoard 出现 `loss/focal`；
- 总损失量级与纯 MSE 接近（权重已归一化到均值 1 + 裁剪上限，量级不会漂移）。
