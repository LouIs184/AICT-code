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


class UNet2d(nn.Module):
    """Standard U-Net with residual output."""

    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super().__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl
        self.model_chl = model_chl

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
        return F.relu(x + conv6)


class UNet2dDeepSuperviser(nn.Module):
    """
    U-Net with intermediate outputs for deep supervision.

    Returns a tuple: (conv5, up4, up3, up2, final_output)
    where each intermediate output can be supervised separately.
    """

    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super().__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl
        self.model_chl = model_chl

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
        res_out = F.relu(x + conv6)        # final output
        return conv5, up4, up3, up2, res_out


class DenseUNet2d(nn.Module):
    """
    U-Net with densely-connected encoder blocks and residual decoder blocks.

    This is the primary model used in the original AICT project.
    """

    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super().__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl
        self.model_chl = model_chl

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
        return F.relu(x + conv6)
