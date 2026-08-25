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
