"""
Data augmentation utilities: MixUp and CutMix.

Original implementations from the AICT project utils.py.
"""

import torch
import numpy as np


class MixUp_AUG:
    """MixUp augmentation for paired (gt, noisy) images."""

    def __init__(self, alpha=1.2):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([alpha]), torch.tensor([alpha])
        )

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda()

        rgb_gt = lam * rgb_gt + (1 - lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1 - lam) * rgb_noisy2

        return rgb_gt, rgb_noisy


def rand_bbox(size, lam):
    """Generate a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix_AUG:
    """CutMix augmentation: replace a rectangular region with a sample from another image."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def aug(self, rgb_gt, rgb_noisy):
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(rgb_gt.size(0))
        bbx1, bby1, bbx2, bby2 = rand_bbox(rgb_gt.size(), lam)

        rgb_gt[:, :, bbx1:bbx2, bby1:bby2] = rgb_gt[rand_index, :, bbx1:bbx2, bby1:bby2]
        rgb_noisy[:, :, bbx1:bbx2, bby1:bby2] = rgb_noisy[rand_index, :, bbx1:bbx2, bby1:bby2]

        return rgb_gt, rgb_noisy


class CutMix_AUG_V2:
    """CutMix V2: per-sample random masks (each sample in batch gets different cut region)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = np.random.beta(self.alpha, self.alpha, bs)
        mask = np.zeros(rgb_gt.size(), dtype=np.float32)
        for i in range(bs):
            bbx1, bby1, bbx2, bby2 = rand_bbox(rgb_gt.size(), lam[i])
            mask[i, :, bbx1:bbx2, bby1:bby2] = 1

        mask = torch.FloatTensor(mask).cuda(rgb_gt.get_device())

        rgb_gt = rgb_gt * (1 - mask) + rgb_gt2 * mask
        rgb_noisy = rgb_noisy * (1 - mask) + rgb_noisy2 * mask

        return rgb_gt, rgb_noisy
