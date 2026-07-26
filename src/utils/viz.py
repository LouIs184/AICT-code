"""
Visualization / image-processing utilities: Gaussian smoothing and patch extraction.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def gaussian_smooth(input_tensor, kernel_size=7, sigma=3.5):
    """
    Apply Gaussian smoothing, then concatenate [input, low, high].

    Args:
        input_tensor: (B, C, H, W) tensor on CUDA.
        kernel_size: size of Gaussian kernel.
        sigma: Gaussian sigma.

    Returns:
        Tensor of shape (B, 3*C, H, W) — [input, low_freq, high_freq].
    """
    gauss_kernel = np.float32(
        np.multiply(
            cv2.getGaussianKernel(kernel_size, sigma),
            np.transpose(cv2.getGaussianKernel(kernel_size, sigma)),
        )
    )
    kernel = gauss_kernel[np.newaxis, np.newaxis, ...]
    kernel = torch.FloatTensor(kernel).cuda(input_tensor.get_device())

    low = F.conv2d(input_tensor, kernel, padding=(kernel_size - 1) // 2)
    high = input_tensor - low

    return torch.cat([input_tensor, low, high], dim=1)


def extract_patches_online(tensor, num=2):
    """
    Extract non-overlapping patches from a tensor by splitting along spatial dims.

    Args:
        tensor: 5D (B, C, D, H, W) or 4D (B, C, H, W).
        num: number of splits per spatial dimension → num^2 (2D) or num^3 (3D) patches.

    Returns:
        Stacked patches tensor.
    """
    if tensor.ndim == 5:
        # 3D volume: split along H and W
        split_w = torch.chunk(tensor, chunks=num, dim=3)
        stack_w = torch.reshape(
            torch.stack(split_w, dim=0),
            [num * tensor.shape[0], tensor.shape[1],
             tensor.shape[2], tensor.shape[3] // num, tensor.shape[4]],
        )
        split_h = torch.chunk(stack_w, chunks=num, dim=4)
        stack_h = torch.reshape(
            torch.stack(split_h, dim=0),
            [num * num * tensor.shape[0], tensor.shape[1],
             tensor.shape[2], tensor.shape[3] // num, tensor.shape[4] // num],
        )
        return stack_h

    elif tensor.ndim == 4:
        # 2D image: split along H and W
        split_w = torch.chunk(tensor, chunks=num, dim=2)
        stack_w = torch.reshape(
            torch.stack(split_w, dim=0),
            [num * tensor.shape[0], tensor.shape[1],
             tensor.shape[2] // num, tensor.shape[3]],
        )
        split_h = torch.chunk(stack_w, chunks=num, dim=3)
        stack_h = torch.reshape(
            torch.stack(split_h, dim=0),
            [num * num * tensor.shape[0], tensor.shape[1],
             tensor.shape[2] // num, tensor.shape[3] // num],
        )
        return stack_h

    else:
        raise ValueError(
            "Expected tensor with ndim == 4 or 5, "
            f"got {tensor.ndim}."
        )
