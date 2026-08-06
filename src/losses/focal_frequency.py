"""
Focal Frequency Loss（频域焦点损失）。

参考论文：Focal Frequency Loss for Image Reconstruction and Synthesis (ICCV 2021)
<https://arxiv.org/pdf/2012.12821.pdf>。算法与官方实现
（limingcv/Focal-Frequency-Loss，<https://github.com/limingcv/Focal-Frequency-Loss>）
完全一致，仅针对本项目环境（torch>=2.0）移除了 torch.fft 的版本分支判断。

原理：在频域（2D DFT）中比较预测与真值的频谱（实部/虚部）。频谱权重矩阵由当前欧氏
距离动态生成——误差大的频率分量（"难"频率）获得更高权重，误差小的"易"频率被降权，
从而引导模型优先修复频域上的难点。与空间域损失（如 mse/charbonnier）互补，建议组合使用。

约定：
- 权重矩阵必须 detach()，不参与梯度回传；
- 权重矩阵归一化到 [0, 1]，loss 量级由频谱欧氏距离决定；
- patch_factor > 1 时把图像切成子块分别做 FFT（patch-based FFL），默认 1；
- ave_spectrum 使用 minibatch 平均频谱，batch_matrix 用 batch 级统计归一化权重。
"""

import torch
import torch.nn as nn
import torch.fft


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss（ICCV 2021）。

    Args:
        loss_weight (float): 整体缩放系数。Default: 1.0。
        alpha (float): 频谱权重矩阵的缩放指数 alpha。Default: 1.0。
        patch_factor (int): 图像切块因子，用于 patch-based FFL。Default: 1。
        ave_spectrum (bool): 是否使用 minibatch 平均频谱。Default: False。
        log_matrix (bool): 是否对频谱权重矩阵取对数调整。Default: False。
        batch_matrix (bool): 是否用 batch 级统计归一化权重矩阵。Default: False。
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1,
                 ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        """将图像张量 (N, C, H, W) 转成频域张量 (N*p^2, C, h/p, w/p, 2)。"""
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            "Patch factor should be divisible by image height and width")

        # 切块
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(
                    x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
                )
        y = torch.stack(patch_list, 1)      # (N, p^2, C, h/p, w/p)

        # 2D DFT（复→复，正交归一化），实部/虚部分别存最后一维
        freq = torch.fft.fft2(y, norm="ortho")
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # 频谱权重矩阵：优先给外部传入的固定矩阵，否则在线动态计算
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                # 逐样本归一化（对最后两个空间维求 max）
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            "The values of spectrum weight matrix should be in the range [0, 1], "
            "but got Min: %.10f Max: %.10f" % (weight_matrix.min().item(), weight_matrix.max().item())
        )

        # 频谱（平方）欧氏距离
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # 动态频谱加权（Hadamard 积）
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """计算 focal frequency loss。

        Args:
            pred (torch.Tensor): of shape (N, C, H, W)。预测张量。
            target (torch.Tensor): of shape (N, C, H, W)。目标张量。
            matrix (torch.Tensor, optional): 元素级频谱权重矩阵。
                Default: None（设为 None 时在线动态计算）。
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # 是否使用 minibatch 平均频谱
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
