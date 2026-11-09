"""
SSIM (Structural Similarity Index Measure) for loss / metric.

Adapted from the original AICT utils.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True,
          data_range=None):
    """Core SSIM computation.

    ``data_range``:
      * ``None`` — legacy behavior: jointly min-max normalize the two images
        to [0, 1]. The effective dynamic range then depends on the prediction
        (and on other samples in the batch), so metrics are not comparable
        across models.
      * a number (e.g. ``1.0``) — inputs are already in a fixed normalized
        range; do not renormalize. Use this in the normalized projection
        domain so SSIM has a fixed, model-independent denominator.
    """
    if data_range is None:
        # Legacy behavior: normalize both images to [0, 1].
        min_val = torch.min(torch.cat([img1, img2], 1))
        max_val = torch.max(torch.cat([img1, img2], 1))
        img1 = (img1 - min_val) / (max_val - min_val + 1e-8)
        img2 = (img2 - min_val) / (max_val - min_val + 1e-8)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
    else:
        # Fixed data range (no joint renormalization).
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    """
    SSIM loss module.

    Can be used as a loss function (1 - SSIM) by setting ``as_loss=True``.
    """

    def __init__(self, window_size=11, size_average=True, as_loss=False,
                 data_range=None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.as_loss = as_loss
        self.data_range = data_range
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        value = _ssim(img1, img2, window, self.window_size, channel,
                      self.size_average, self.data_range)
        if self.as_loss:
            return 1 - value
        return value


def ssim(img1, img2, window_size=11, size_average=True, data_range=None):
    """Functional SSIM."""
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average,
                 data_range)
