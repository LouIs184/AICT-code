"""
VGG-based perceptual loss for image reconstruction.

Uses a pre-trained VGG16 network as a fixed feature extractor
and computes MSE between intermediate feature maps of the
generated and reference images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class vgg_feature_extractor(nn.Module):
    """
    VGG16 backbone truncated at layer 27 (before final pooling).

    Outputs features at 5 intermediate layers:
        relu1_2, relu2_2, relu3_3, relu4_3, relu5_3

    If input is single-channel, it is replicated to 3 channels.
    """

    def __init__(self):
        super().__init__()
        vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg_layers = nn.Sequential(*list(vgg_model.features.children())[:27])

    def forward(self, img):
        if img.shape[1] != 3:
            img = img.repeat(1, 3, 1, 1)

        # Store output before each block
        out = [None] * (len(self.vgg_layers) + 1)
        out[0] = img

        for i in range(len(self.vgg_layers)):
            out[i + 1] = self.vgg_layers[i](out[i])

        # Return feature maps at: relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        # Indices in truncated VGG: 2, 7, 12, 19, 26
        return out[2], out[7], out[12], out[19], out[26]


def vgg_loss_calc(x, y, vgg_extractor):
    """
    Compute perceptual loss as MSE between VGG feature maps.

    Args:
        x: generated image (B, C, H, W)
        y: reference image (B, C, H, W)
        vgg_extractor: vgg_feature_extractor instance

    Returns:
        Scalar VGG perceptual loss.
    """
    xf1, xf2, xf3, xf4, xf5 = vgg_extractor(x)
    yf1, yf2, yf3, yf4, yf5 = vgg_extractor(y)

    vgg_loss = (F.mse_loss(xf1, yf1) +
                F.mse_loss(xf2, yf2) +
                F.mse_loss(xf3, yf3) +
                F.mse_loss(xf4, yf4) +
                F.mse_loss(xf5, yf5))
    return vgg_loss
