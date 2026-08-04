"""
自适应焦点权重 MSE（Focal Loss for pixel regression）。

依据 docs/只学习简单像素矫正/HARD_REGION_OPTIMIZATION_GUIDE.md 4.3 实现。

原理：w = |pred - target|^gamma，按当前误差动态加权——模型哪里错得多，哪里就
获得更高权重。比静态权重更"主动"，但 gamma 过大容易震荡；可用 ``ema_decay``
对误差图做滑动平均，抑制帧间抖动。

约定：
- 权重图必须 detach()，不参与梯度回传，否则等价于更高次幂损失、梯度易爆；
- 权重归一化到均值=1 并裁剪上限，避免 loss 量级漂移；
- 建议与少量普通 MSE 组合使用（``mse_weight`` 保底），不要全替换。
"""

import torch
import torch.nn as nn


class FocalMSE(nn.Module):
    """MSE + 自适应焦点权重 w = |pred - target|^gamma（4.3）。"""

    def __init__(self, gamma=1.0, wmax=20.0, ema_decay=0.0):
        super().__init__()
        self.gamma = gamma
        self.wmax = wmax
        self.ema_decay = ema_decay
        # EMA 误差图（None 表示尚未初始化）；用普通属性而非 buffer，避免被
        # persist 到 checkpoint / 随 module 移动，forward 时与 err 同设备。
        self.ema_err = None

    def forward(self, pred, target):
        err = (pred - target).abs().detach()      # 关键：detach，否则会放大梯度

        if self.ema_decay > 0:
            if self.ema_err is None or self.ema_err.shape != err.shape:
                self.ema_err = err.clone()
            else:
                self.ema_err.mul_(self.ema_decay).add_(err, alpha=1 - self.ema_decay)
            err = self.ema_err

        w = (err + 1e-6) ** self.gamma
        w = w / (w.mean() + 1e-8)                 # 归一化到均值=1，保持 loss 量级稳定
        w = torch.clamp(w, max=self.wmax)         # 防止单像素主导
        return (w * (pred - target) ** 2).mean()
