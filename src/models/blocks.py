"""
Basic building blocks for U-Net variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu2d(nn.Module):
    """Convolution + optional BatchNorm + optional ReLU."""

    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1,
                 stride=1, groups=1, is_bn=True, is_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              stride=stride, dilation=dilation,
                              groups=groups, bias=True)
        self.bn = nn.BatchNorm2d(out_chl, eps=1e-4) if is_bn else None
        self.relu = nn.ReLU(inplace=True) if is_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# ---------------------------------------------------------------------------
# Standard U-Net encoder / decoder blocks
# ---------------------------------------------------------------------------

class StackEncoder(nn.Module):
    """Two conv-blocks followed by max-pool."""

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super().__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2,
                                padding=0, ceil_mode=True)
        return conv_out, down_out


class StackDecoder(nn.Module):
    """Upsample → concat skip → two conv-blocks."""

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size),
        )

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out = F.interpolate(up_in, size=(H, W), mode="bilinear",
                               align_corners=False)
        conv_out = self.conv(torch.cat([up_out, conv_res], dim=1))
        return conv_out


# ---------------------------------------------------------------------------
# Dense-style encoder & decoder blocks
# ---------------------------------------------------------------------------

class StackDenseEncoder2d(nn.Module):
    """Dense-style encoder: 3 convs with dense skip connections."""

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super().__init__()
        self.conv1 = ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size)
        self.conv2 = ConvBnRelu2d(in_chl + out_chl, out_chl,
                                  kernel_size=kernel_size)
        self.conv3 = ConvBnRelu2d(in_chl + 2 * out_chl, out_chl,
                                  kernel_size=kernel_size, is_relu=False)
        self.convx = None
        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1,
                                      is_bn=False, is_relu=False)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], dim=1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], dim=1))
        if self.convx is None:
            convx = F.relu(conv3 + x)
        else:
            convx = F.relu(conv3 + self.convx(x))
        down_out = F.max_pool2d(convx, kernel_size=2, stride=2,
                                padding=0, ceil_mode=True)
        return convx, down_out


class StackDenseBlock2d(nn.Module):
    """Dense-style bottleneck block (no down-sampling)."""

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super().__init__()
        self.conv1 = ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size)
        self.conv2 = ConvBnRelu2d(in_chl + out_chl, out_chl,
                                  kernel_size=kernel_size)
        self.conv3 = ConvBnRelu2d(in_chl + 2 * out_chl, out_chl,
                                  kernel_size=kernel_size, is_relu=False)
        self.convx = None
        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1,
                                      is_bn=False, is_relu=False)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], dim=1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], dim=1))
        if self.convx is None:
            convx = F.relu(conv3 + x)
        else:
            convx = F.relu(conv3 + self.convx(x))
        return convx


class StackResDecoder2d(nn.Module):
    """Residual-style decoder: upsample → concat → conv + residual."""

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super().__init__()
        self.conv1 = ConvBnRelu2d(in_chl + out_chl, out_chl,
                                  kernel_size=kernel_size)
        self.conv2 = ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size,
                                  is_relu=False)
        self.convx = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=1,
                                  is_bn=False, is_relu=False)

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out = F.interpolate(up_in, size=(H, W), mode="bilinear",
                               align_corners=False)
        cat = torch.cat([up_out, conv_res], dim=1)
        conv1 = self.conv1(cat)
        conv2 = self.conv2(conv1)
        convx = F.relu(conv2 + self.convx(cat))
        return convx
