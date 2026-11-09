"""正向相位传播算子（物理一致性损失用）。

    post-log 线积分 x
      --(delta/beta 为常数)-->  复场 E0 = exp(-x/2 - i*kappa*x/2)
      --角谱传播（与 Xufft2Ver2 同频点）-->  复场
      --|.|^2 -->  强度 --> -log -->  post-log

与原仿真保持一致的两个关键点：

* 频率格点带半格偏移（``vk = (n - (N-1)/2) / N``），不是 numpy 的 ``fftfreq``；
  照搬 ``Xufft2Ver2``，否则在低菲涅尔数时等效于半像素错位。
* 传播核 ``H = exp(-i*pi*f_px^2 / NF)``，只依赖菲涅尔数
  ``NF = a_eff^2 / (lambda * z_eff)``；不做上采样/下采样，直接在输入网格上传播。

数值上用菲涅尔形式的核：精确角谱 ``exp(2*pi*i*z*sqrt(1/lambda^2-f^2))`` 的相位可达
1e11 rad，float32 精度不足；两者对 |E|^2 只差一个全局常数相位和 1e-9 量级的修正。

本文件从 ``~/restormer/basicsr/models/physics_prop.py`` 移植，数值细节保持不变。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LAM = 1.24 / 50 * 1e-6      # 50 keV，单位 mm
KAPPA = 2058.2544           # delta / beta
SID = 1e6                   # 源到探测器距离，mm


def compute_nf(a, z, lam=LAM, sid=SID):
    """由光斑尺寸 a (mm) 和传播距离 z (mm) 计算菲涅尔数。"""
    m = 1.0 + float(z) / float(sid)
    a_eff = float(a) / m
    z_eff = float(z) / m
    return a_eff ** 2 / (float(lam) * z_eff)


def xufft(x, vecx, vecy, fwd):
    """与 MATLAB ``Xufft2Ver2`` 等价的中心化 FFT。

    Args:
        x (Tensor): (B, H, W) complex。
        vecx (Tensor): 长度 H 的坐标（正变换）或频率（逆变换）向量。
        vecy (Tensor): 长度 W 的同上。
        fwd (bool): True 为空间 -> 频率。

    Returns:
        tuple: (变换结果, 频率向量 vkx, vyk)。
    """
    n1d, n2d = x.shape[-2], x.shape[-1]
    dx = float(vecx[1] - vecx[0])
    dy = float(vecy[1] - vecy[0])
    x0 = float(vecx[0])
    y0 = float(vecy[0])

    kx0 = (1 - n1d) / 2 / (n1d * dx)
    ky0 = (1 - n2d) / 2 / (n2d * dy)

    n1 = torch.arange(n1d, device=x.device, dtype=torch.float64)
    n2 = torch.arange(n2d, device=x.device, dtype=torch.float64)
    s = -1j if fwd else 1j

    mod = (torch.exp(s * 2 * math.pi * n1 * dx * kx0)[:, None]
           * torch.exp(s * 2 * math.pi * n2 * dy * ky0)[None, :]).to(x.dtype)
    if fwd:
        temp = torch.fft.fft2(x * mod) * (dx * dy)
    else:
        temp = torch.fft.ifft2(x * mod) * (n1d * n2d * dx * dy)

    vk1 = n1 / (n1d * dx) + kx0
    vk2 = n2 / (n2d * dy) + ky0
    ph = (torch.exp(s * 2 * math.pi * vk1 * x0)[:, None]
          * torch.exp(s * 2 * math.pi * vk2 * y0)[None, :]).to(temp.dtype)
    return temp * ph, vk1, vk2


def phys_forward(x, nf):
    """把 post-log 线积分正向传播成测量域的 post-log。

    Args:
        x (Tensor): (B, 1, H, W)，物理单位（无量纲线积分）。
        nf (Tensor): (B,) 或 (B, 1, 1)，每个样本的菲涅尔数。

    Returns:
        Tensor: (B, 1, H, W)，``-log|P(E0)|^2``。
    """
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f'x should be (B,1,H,W), got {tuple(x.shape)}')
    b, _, h, w = x.shape

    # 用 float64 算复场：相位约 200 rad，-log 在暗条纹处对数值误差敏感
    xf = x[:, 0].double()
    e0 = torch.complex(-0.5 * xf, -0.5 * KAPPA * xf).exp()   # (B,H,W)

    # 像素单位（d = 1）：坐标范围 -(N-1)/2 ... +(N-1)/2
    vy = ((torch.arange(h, device=x.device, dtype=torch.float64) + 1)
          - (1 + h) / 2)
    vx = ((torch.arange(w, device=x.device, dtype=torch.float64) + 1)
          - (1 + w) / 2)

    y, vkx, vky = xufft(e0, vy, vx, True)
    f2 = vkx[:, None] ** 2 + vky[None, :] ** 2            # cycles / pixel

    nf = nf.to(device=x.device, dtype=torch.float64).reshape(-1, 1, 1)
    if nf.shape[0] != b:
        nf = nf.expand(b, 1, 1)
    hk = torch.exp(-1j * math.pi * (f2[None] / nf)).to(y.dtype)

    u, _, _ = xufft(y * hk, vkx, vky, False)

    inten = u.real ** 2 + u.imag ** 2
    return -torch.log(inten.clamp_min(1e-8)).unsqueeze(1)


class PhysConsistency(nn.Module):
    """在归一化域之间做正向传播。

    输入 pred、目标 meas 都用 yml 里的 ``proj_norm_clip_max`` 归一化；
    内部乘回该尺度换成物理单位，算完再除回去，方便与 meas 直接比较。

    Args:
        clip_max (float): 归一化上界。
        pad (int): 传播前四周 reflect padding 的像素数，减轻周期边界效应。
    """

    def __init__(self, clip_max, pad=8):
        super().__init__()
        if clip_max is None or clip_max <= 0:
            raise ValueError(f'clip_max must be positive, got {clip_max}')
        self.clip_max = float(clip_max)
        self.pad = int(pad)

    def forward(self, pred, nf):
        x = pred * self.clip_max
        p = min(self.pad, x.shape[-2] - 1, x.shape[-1] - 1)
        if p > 0:
            x = F.pad(x, (p, p, p, p), mode='reflect')
        y = phys_forward(x, nf)
        if p > 0:
            y = y[..., p:-p, p:-p]
        return y / self.clip_max
