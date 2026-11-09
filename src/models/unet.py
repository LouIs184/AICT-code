"""
U-Net variants for CT image reconstruction.

All models accept single-channel 2D inputs and produce single-channel 2D outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConvBnRelu2d,
    StackEncoder,
    StackDecoder,
    StackDenseEncoder2d,
    StackDenseBlock2d,
    StackResDecoder2d,
)


def _validate_output_options(residual_output, output_activation, residual_init):
    """Fail fast on invalid output-switch combinations."""
    if output_activation not in ("relu", "none"):
        raise ValueError(
            f"output_activation must be 'relu' or 'none', got {output_activation!r}"
        )
    if residual_init and not residual_output:
        raise ValueError("residual_init=True requires residual_output=True")


class UNet2d(nn.Module):
    """Standard U-Net with an optional global input-to-output residual."""

    def __init__(self, in_chl=1, out_chl=1, model_chl=32, artifact_output=False,
                 residual_output=True, output_activation="relu",
                 residual_init=False):
        super().__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.artifact_output = artifact_output
        self.residual_output = residual_output
        self.output_activation = output_activation
        self.residual_init = residual_init
        _validate_output_options(residual_output, output_activation, residual_init)

        self.begin = nn.Sequential(
            ConvBnRelu2d(in_chl, model_chl, kernel_size=3, stride=1, is_bn=False)
        )
        self.down1 = StackEncoder(model_chl, model_chl)                # 256
        self.down2 = StackEncoder(model_chl, model_chl * 2)            # 128
        self.down3 = StackEncoder(model_chl * 2, model_chl * 4)        # 64
        self.down4 = StackEncoder(model_chl * 4, model_chl * 8)        # 32

        self.center = nn.Sequential(
            ConvBnRelu2d(model_chl * 8, model_chl * 16, kernel_size=3),
            ConvBnRelu2d(model_chl * 16, model_chl * 16, kernel_size=3),
        )

        self.up4 = StackDecoder(model_chl * 16, model_chl * 8)
        self.up3 = StackDecoder(model_chl * 8, model_chl * 4)
        self.up2 = StackDecoder(model_chl * 4, model_chl * 2)
        self.up1 = StackDecoder(model_chl * 2, model_chl)

        self.end = nn.Sequential(
            ConvBnRelu2d(model_chl, out_chl, kernel_size=1, stride=1,
                         is_bn=False, is_relu=False)
        )

        if self.residual_init and self.residual_output:
            # 近零初始化输出头：训练起点 f(x)≈0，整体输出≈输入
            conv = self.end[0].conv
            nn.init.normal_(conv.weight, mean=0.0, std=0.001)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

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
        out = x + conv6 if self.residual_output else conv6
        if self.output_activation == "relu":
            out = F.relu(out)
        return out


class UNet2dDeepSuperviser(nn.Module):
    """
    U-Net with intermediate outputs for deep supervision.

    Returns a tuple: (conv5, up4, up3, up2, final_output)
    where each intermediate output can be supervised separately.
    """

    def __init__(self, in_chl=1, out_chl=1, model_chl=32, artifact_output=False,
                 residual_output=True, output_activation="relu",
                 residual_init=False):
        super().__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.artifact_output = artifact_output
        self.residual_output = residual_output
        self.output_activation = output_activation
        self.residual_init = residual_init
        _validate_output_options(residual_output, output_activation, residual_init)

        self.begin = nn.Sequential(
            ConvBnRelu2d(in_chl, model_chl, kernel_size=3, stride=1, is_bn=False)
        )
        self.down1 = StackEncoder(model_chl, model_chl)
        self.down2 = StackEncoder(model_chl, model_chl * 2)
        self.down3 = StackEncoder(model_chl * 2, model_chl * 4)
        self.down4 = StackEncoder(model_chl * 4, model_chl * 8)

        self.center = nn.Sequential(
            ConvBnRelu2d(model_chl * 8, model_chl * 16, kernel_size=3),
            ConvBnRelu2d(model_chl * 16, model_chl * 16, kernel_size=3),
        )

        self.up4 = StackDecoder(model_chl * 16, model_chl * 8)
        self.up3 = StackDecoder(model_chl * 8, model_chl * 4)
        self.up2 = StackDecoder(model_chl * 4, model_chl * 2)
        self.up1 = StackDecoder(model_chl * 2, model_chl)

        self.end = nn.Sequential(
            ConvBnRelu2d(model_chl, out_chl, kernel_size=1, stride=1,
                         is_bn=False, is_relu=False)
        )

        if self.residual_init and self.residual_output:
            conv = self.end[0].conv
            nn.init.normal_(conv.weight, mean=0.0, std=0.001)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)       # intermediate supervision
        up3 = self.up3(up4, conv3)         # intermediate supervision
        up2 = self.up2(up3, conv2)         # intermediate supervision
        up1 = self.up1(up2, conv1)         # intermediate supervision
        conv6 = self.end(up1)
        res_out = x + conv6 if self.residual_output else conv6
        if self.output_activation == "relu":
            res_out = F.relu(res_out)
        return conv5, up4, up3, up2, res_out


class DenseUNet2d(nn.Module):
    """
    U-Net with densely-connected encoder blocks and residual decoder blocks.

    This is the primary model used in the original AICT project.
    """

    def __init__(self, in_chl=1, out_chl=1, model_chl=32, artifact_output=False,
                 residual_output=True, output_activation="relu",
                 residual_init=False):
        super().__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.artifact_output = artifact_output
        self.residual_output = residual_output
        self.output_activation = output_activation
        self.residual_init = residual_init
        _validate_output_options(residual_output, output_activation, residual_init)

        self.begin = nn.Sequential(
            ConvBnRelu2d(in_chl, model_chl, kernel_size=3, stride=1, is_bn=False)
        )
        self.down1 = StackDenseEncoder2d(model_chl, model_chl)
        self.down2 = StackDenseEncoder2d(model_chl, model_chl * 2)
        self.down3 = StackDenseEncoder2d(model_chl * 2, model_chl * 4)
        self.down4 = StackDenseEncoder2d(model_chl * 4, model_chl * 8)

        self.center = StackDenseBlock2d(model_chl * 8, model_chl * 16)

        self.up4 = StackResDecoder2d(model_chl * 16, model_chl * 8)
        self.up3 = StackResDecoder2d(model_chl * 8, model_chl * 4)
        self.up2 = StackResDecoder2d(model_chl * 4, model_chl * 2)
        self.up1 = StackResDecoder2d(model_chl * 2, model_chl)

        self.end = nn.Sequential(
            ConvBnRelu2d(model_chl, out_chl, kernel_size=1, stride=1,
                         is_bn=False, is_relu=False)
        )

        if artifact_output:
            # 恒等起点：输出层近零初始化 → 训练一开始 artifact≈0，clean≈x
            conv = self.end[0].conv
            nn.init.normal_(conv.weight, mean=0.0, std=0.001)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)
        elif self.residual_init and self.residual_output:
            # 残差模式下的恒等起点：输出头近零初始化
            conv = self.end[0].conv
            nn.init.normal_(conv.weight, mean=0.0, std=0.001)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

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
        out = x + conv6 if self.residual_output else conv6
        if self.output_activation == "relu":
            out = F.relu(out)
        return out
