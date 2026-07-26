"""
Metrics tracking and normalization utilities.
"""

import torch
import math


class AverageMeter:
    """
    Computes and stores the average and current value.

    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalization(tensor):
    """Min-max normalization to [0, 1]."""
    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor) + 1e-8)


def compute_psnr(mse, max_val=1.0):
    """Compute PSNR from MSE value."""
    if mse < 1e-10:
        return float("inf")
    return -10 * math.log10(mse + 1e-8)
