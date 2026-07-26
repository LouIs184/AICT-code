"""
GAN discriminator and WGAN-GP gradient penalty.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable

from .blocks import ConvBnRelu2d


class Discriminator2D(nn.Module):
    """
    Convolutional discriminator for 2D image patches.

    Architecture:
      4 conv layers (stride-2 down-sampling) → adaptive avg pool → linear output.
    """

    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super().__init__()
        self.in_chl = in_chl
        self.out_chl = out_chl
        self.model_chl = model_chl

        self.ConvLayers = nn.Sequential(
            ConvBnRelu2d(in_chl, model_chl, kernel_size=3, stride=1),
            ConvBnRelu2d(model_chl, model_chl, kernel_size=3, stride=2,
                         is_bn=False, is_relu=False),

            ConvBnRelu2d(model_chl, model_chl * 2, kernel_size=3, stride=1),
            ConvBnRelu2d(model_chl * 2, model_chl * 2, kernel_size=3, stride=2,
                         is_bn=False, is_relu=False),

            ConvBnRelu2d(model_chl * 2, model_chl * 4, kernel_size=3, stride=1),
            ConvBnRelu2d(model_chl * 4, model_chl * 4, kernel_size=3, stride=2,
                         is_bn=False, is_relu=False),

            ConvBnRelu2d(model_chl * 4, model_chl * 8, kernel_size=3, stride=1),
            ConvBnRelu2d(model_chl * 8, model_chl * 8, kernel_size=3, stride=2,
                         is_bn=False, is_relu=False),
        )

        self.FCLayer = nn.Sequential(
            nn.Linear(model_chl * 8, out_chl),
        )

    def forward(self, x):
        out = self.ConvLayers(x)
        out = torch.reshape(F.adaptive_avg_pool2d(out, (1, 1)),
                            [out.shape[0], out.shape[1]])
        out = self.FCLayer(out)
        return out


def compute_gradient_penalty(D, real_samples, fake_samples):
    """
    WGAN-GP gradient penalty.

    Args:
        D: discriminator network.
        real_samples: batch of real samples.
        fake_samples: batch of generated samples.

    Returns:
        Scalar gradient penalty loss.
    """
    Tensor = torch.cuda.FloatTensor
    alpha_shape = (real_samples.size(0),) + (1,) * (real_samples.ndim - 1)
    alpha = Tensor(np.random.random(alpha_shape))

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0),
                    requires_grad=False)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
