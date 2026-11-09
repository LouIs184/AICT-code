"""
Unified training engine for CT image reconstruction.

Replaces the 9 duplicated training scripts from the original AICT project.
Handles:
  - Standard / GAN / Deep Supervision training
  - MixUp / CutMix data augmentation
  - Multi-GPU DataParallel
  - TensorBoard logging with image summaries
  - Checkpoint save / resume
"""

from __future__ import annotations

import os
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from src.config import Config, save_config
from src.models import (UNet2d, UNet2dDeepSuperviser, DenseUNet2d,
                        Discriminator2D, PhysConsistency)
from src.losses import CharbonnierLoss, SSIM, vgg_feature_extractor, vgg_loss_calc, TVLoss, FocalMSE, FocalFrequencyLoss
from src.losses import MultiDirectionalSobelLoss, FFTAmplitudeLoss, AngularGradLoss
from src.data import raw_reader, dicom_reader, proj_reader
from src.utils.metrics import AverageMeter, normalization
from src.utils.io_utils import save_model, load_model_model_parallel, find_lastest_file
from src.utils.augments import MixUp_AUG, CutMix_AUG, CutMix_AUG_V2
from src.optimizers import SophiaG

# Model registry
MODEL_MAP = {
    "UNet2d": UNet2d,
    "UNet2dDeepSuperviser": UNet2dDeepSuperviser,
    "DenseUNet2d": DenseUNet2d,
}

# Reader registry
READER_MAP = {
    "raw": raw_reader,
    "dicom": dicom_reader,
    "proj": proj_reader,
}

# Optimizer registry
OPTIMIZER_MAP = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SophiaG": SophiaG,
}


def _set_seed(seed: int):
    """Seed Python / NumPy / PyTorch RNGs before building data + model.

    Paired A/B runs (e.g. residual vs no-residual) rely on identical weight
    initialization and shuffle order, so this must run before DataLoader and
    model construction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(model, cfg):
    """Create optimizer from config."""
    opt_cls = OPTIMIZER_MAP.get(cfg.optimizer.name)
    if opt_cls is None:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")

    if cfg.optimizer.name == "SophiaG":
        return opt_cls(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        return opt_cls(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )


def _build_scheduler(optimizer, cfg):
    """Create LR scheduler from config."""
    name = cfg.scheduler.name
    if name == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size,
                                          gamma=cfg.scheduler.gamma)
    elif name == "MultiStepLR":
        milestones = cfg.scheduler.milestones or [100, 150]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                               gamma=cfg.scheduler.gamma)
    elif name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler.t_max)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


class Trainer:
    """
    Unified trainer for all AICT experiment configurations.

    Usage::

        from src.config import load_config
        from src.trainer import Trainer

        cfg = load_config("experiments/denseunet_phasect.yaml")
        trainer = Trainer(cfg)
        trainer.fit()
    """

    def __init__(self, config: Config, verbose: bool = True):
        self.cfg = config
        self.verbose = verbose
        _set_seed(config.experiment.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cudnn.benchmark = True

        self._build_dataloaders()
        self._build_model()
        self._build_losses()
        self._build_physics()
        self._build_optimizers()
        self._build_schedulers()
        self._build_writer()

        self.current_epoch = 0
        self.global_step = 0

        # Augmentation
        self.augment = None
        aug_name = config.augment.name
        if aug_name == "mixup":
            self.augment = MixUp_AUG(alpha=config.augment.mixup_alpha)
        elif aug_name == "cutmix":
            self.augment = CutMix_AUG()
        elif aug_name == "cutmix_v2":
            self.augment = CutMix_AUG_V2()

        # VGG feature extractor (lazy init)
        self.vgg_extractor = None

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_dataloaders(self):
        dc = self.cfg.data
        reader_cls = READER_MAP.get(dc.reader)
        if reader_cls is None:
            raise ValueError(f"Unknown reader: {dc.reader}")

        reader_kwargs = {}
        if dc.reader == "raw":
            reader_kwargs["hu_min"] = dc.hu_min
            reader_kwargs["hu_max"] = dc.hu_max
        elif dc.reader == "proj":
            reader_kwargs["norm_clip_max"] = dc.norm_clip_max
            reader_kwargs["crop_rows"] = dc.crop_rows
            reader_kwargs["crop_cols"] = dc.crop_cols
            reader_kwargs["view_shape"] = tuple(dc.view_shape)

        train_kwargs = dict(reader_kwargs)
        if dc.reader == "proj":
            # 物理目标与 per-head NF 只训练集需要；验证不反传，省 I/O
            train_kwargs["phys_file"] = dc.phys_file
            train_kwargs["heads_json"] = dc.heads_json
            train_kwargs["clip_meas"] = dc.clip_meas

        self.train_dataset = reader_cls(paired_img_txt=dc.train_txt, **train_kwargs)
        self.valid_dataset = reader_cls(paired_img_txt=dc.valid_txt, **reader_kwargs)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=dc.batch_size,
            num_workers=dc.num_workers, shuffle=dc.shuffle,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=dc.valid_batch_size,
            num_workers=dc.num_workers, shuffle=False,
            pin_memory=True,
        )

    def _build_model(self):
        mc = self.cfg.model
        model_cls = MODEL_MAP.get(mc.name)
        if model_cls is None:
            raise ValueError(f"Unknown model: {mc.name} (available: {list(MODEL_MAP.keys())})")

        model = model_cls(in_chl=mc.in_chl, out_chl=mc.out_chl, model_chl=mc.model_chl,
                          artifact_output=mc.artifact_output,
                          residual_output=mc.residual_output,
                          output_activation=mc.output_activation,
                          residual_init=mc.residual_init)

        # Multi-GPU
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        self.model = model

        # GAN 与伪影残差输出不兼容，提前报错（避免 _train_gan_step 里晦涩的 tuple 崩溃）
        if mc.artifact_output and self.cfg.loss.adv_weight > 0:
            raise ValueError(
                "artifact_output=True 时不支持 GAN（adv_weight>0）。请在 yaml 的 loss: 段设置 adv_weight: 0.0"
            )

        # GAN discriminator
        self.discriminator = None
        if self.cfg.loss.adv_weight > 0:
            self.discriminator = Discriminator2D().to(self.device)

    def _build_physics(self):
        """Build the forward propagation module (optional)."""
        pc = self.cfg.physics
        self.phys = None
        if not pc.enable:
            return
        clip = self.cfg.data.norm_clip_max
        if clip is None or float(clip) <= 0:
            raise ValueError(
                "physics.enable=True requires data.norm_clip_max > 0")
        if pc.target == "file" and not self.cfg.data.phys_file:
            raise ValueError(
                "physics.target='file' requires data.phys_file to be set")
        self.phys = PhysConsistency(float(clip), pc.pad).to(self.device)
        if self.verbose:
            print(f"[Physics] enabled: lambda={pc.loss_weight}, pad={pc.pad}, "
                  f"start_epoch={pc.start_epoch}, warmup={pc.warmup_epochs}, "
                  f"target={pc.target}, target_file={pc.target_file}")

    def _build_losses(self):
        lc = self.cfg.loss
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_charbonnier = CharbonnierLoss(eps=lc.charbonnier_eps)
        # AICT readers normalize to [0, 1]; a fixed range keeps SSIM
        # independent of the prediction's own extrema.
        self.criterion_ssim = SSIM(data_range=1.0)
        self.criterion_tv = TVLoss()
        self.criterion_fft_amp = FFTAmplitudeLoss().to(self.device)
        self.criterion_angular = AngularGradLoss().to(self.device)
        self.criterion_sobel = MultiDirectionalSobelLoss().to(self.device)

        # 自适应焦点权重（4.3），hard_weight>0 时启用
        self.criterion_focal = None
        if lc.hard_weight > 0:
            self.criterion_focal = FocalMSE(
                gamma=lc.hard_gamma, wmax=lc.hard_wmax, ema_decay=lc.hard_ema_decay
            ).to(self.device)

        # Focal Frequency Loss（ICCV 2021），ffl_weight>0 时启用
        self.criterion_ffl = None
        if lc.ffl_weight > 0:
            self.criterion_ffl = FocalFrequencyLoss(
                alpha=lc.ffl_alpha, patch_factor=lc.ffl_patch_factor,
                ave_spectrum=lc.ffl_ave_spectrum, log_matrix=lc.ffl_log_matrix,
                batch_matrix=lc.ffl_batch_matrix,
            ).to(self.device)

    def _build_optimizers(self):
        self.optimizer = _build_optimizer(self.model, self.cfg)

        if self.discriminator is not None:
            gc = self.cfg.gan
            self.optimizer_d = optim.Adam(
                self.discriminator.parameters(),
                lr=gc.d_lr,
                betas=tuple(gc.d_betas),
            )
        else:
            self.optimizer_d = None

    def _build_schedulers(self):
        self.scheduler = _build_scheduler(self.optimizer, self.cfg)
        self.scheduler_d = (
            _build_scheduler(self.optimizer_d, self.cfg)
            if self.optimizer_d is not None else None
        )

    def _build_writer(self):
        tc = self.cfg.training
        self.log_dir = os.path.join(tc.log_dir, self.cfg.experiment.name)
        self.save_dir = os.path.join(tc.save_dir, "checkpoints")

        now = datetime.now().strftime("%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(tc.log_dir, self.cfg.experiment.name, now)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # 保存本次训练生效的完整配置（含 CLI 覆盖），便于回溯。
        # log_dir 下按时间戳隔离，不会被后续训练覆盖；save_dir 下是最近一次运行的配置。
        save_config(self.cfg, os.path.join(self.log_dir, "config.yaml"))
        os.makedirs(self.save_dir, exist_ok=True)
        save_config(self.cfg, os.path.join(self.save_dir, "config.yaml"))

        if self.verbose:
            print(f"Logging to: {self.log_dir}")
            print(f"Checkpoints: {self.save_dir}")

    def _get_vgg_extractor(self):
        if self.vgg_extractor is None:
            self.vgg_extractor = vgg_feature_extractor().to(self.device)
            self.vgg_extractor.eval()
        return self.vgg_extractor

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(self, predict, target, pred_artifact=None, gt_artifact=None):
        """Compute combined loss from configured weights."""
        lc = self.cfg.loss
        loss = 0.0
        losses = {}

        if lc.mse_weight > 0:
            l_mse = self.criterion_mse(predict, target)
            losses["mse"] = l_mse
            loss += lc.mse_weight * l_mse

        # ---- TMCode 移植：L1 / 伪影残差 L1 / FFT 幅值 / 角度梯度 / 多向 Sobel ----
        if lc.l1_weight > 0:
            l_l1 = self.criterion_l1(predict, target)
            losses["l1"] = l_l1
            loss += lc.l1_weight * l_l1

        if lc.artifact_weight > 0 and pred_artifact is not None and gt_artifact is not None:
            l_art = self.criterion_l1(pred_artifact, gt_artifact)
            losses["artifact"] = l_art
            loss += lc.artifact_weight * l_art

        if lc.fft_amp_weight > 0:
            l_fft = self.criterion_fft_amp(predict, target)
            losses["fft_amp"] = l_fft
            loss += lc.fft_amp_weight * l_fft

        if lc.angular_weight > 0:
            l_ang = self.criterion_angular(predict, target)
            losses["angular"] = l_ang
            loss += lc.angular_weight * l_ang

        if lc.sobel_weight > 0:
            l_sob = self.criterion_sobel(predict, target)
            losses["sobel"] = l_sob
            loss += lc.sobel_weight * l_sob

        if lc.charbonnier_weight > 0:
            l_cl = self.criterion_charbonnier(predict, target)
            losses["charbonnier"] = l_cl
            loss += lc.charbonnier_weight * l_cl

        if lc.ssim_weight > 0:
            l_ssim = 1 - self.criterion_ssim(predict.clamp(0, 1), target)
            losses["ssim"] = l_ssim
            loss += lc.ssim_weight * l_ssim

        if lc.tv_weight > 0:
            l_tv = self.criterion_tv(predict)
            losses["tv"] = l_tv
            loss += lc.tv_weight * l_tv

        if lc.vgg_weight > 0:
            vgg_ext = self._get_vgg_extractor()
            l_vgg = vgg_loss_calc(predict, target, vgg_ext)
            losses["vgg"] = l_vgg
            loss += lc.vgg_weight * l_vgg

        # 自适应焦点权重（4.3）：当前误差大的像素更值钱
        if lc.hard_weight > 0 and self.criterion_focal is not None:
            l_focal = self.criterion_focal(predict, target)
            losses["focal"] = l_focal
            loss += lc.hard_weight * l_focal

        # Focal Frequency Loss（ICCV 2021）：频域误差动态加权，关注"难"频率
        if lc.ffl_weight > 0 and self.criterion_ffl is not None:
            l_ffl = self.criterion_ffl(predict, target)
            losses["ffl"] = l_ffl
            loss += lc.ffl_weight * l_ffl

        return loss, losses

    def _compute_phys_loss(self, pred, gt, nf, meas, epoch):
        """物理一致性损失：A(pred) 与传播目标 M 的相对误差 ρ。

        M 优先用 dataloader 给的 ``meas``（varNF 的 proj_pbi.raw）；
        没有时退回在线 ``A(gt)``。
        """
        pc = self.cfg.physics
        if self.phys is None or epoch < pc.start_epoch:
            return None, None

        if nf is None:
            nf = torch.full((pred.size(0),), float(pc.nf),
                            device=pred.device, dtype=pred.dtype)
        else:
            nf = nf.to(pred.device).reshape(-1).to(pred.dtype)

        a_pred = self.phys(pred, nf)
        if meas is None:
            with torch.no_grad():
                m = self.phys(gt, nf)
        else:
            m = meas.to(pred.device, dtype=pred.dtype)

        if pc.normalize_by_signal:
            with torch.no_grad():
                sig = (m - gt).pow(2).flatten(1).mean(1).sqrt().clamp_min(1e-6)
            rho = (a_pred - m).pow(2).flatten(1).mean(1).sqrt() / sig
            if pc.rho_clip > 0:
                rho = rho.clamp(max=pc.rho_clip)
            l_phys = rho.mean()
        else:
            l_phys = F.l1_loss(a_pred, m)
            rho = l_phys.detach()

        return l_phys, rho.mean()

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch):
        cfg = self.cfg
        model = self.model
        model.train()

        if self.discriminator is not None:
            self.discriminator.train()

        batch_time = AverageMeter()
        loss_meters = {}                # 每个损失分项一个 meter（mse/focal/...）
        end = time.time()

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}",
                     leave=False, disable=not self.verbose)

        for step, data in enumerate(pbar):
            ldct = data["ldct"].to(self.device, non_blocking=True)
            rdct = data["rdct"].to(self.device, non_blocking=True)

            # ---- Data augmentation ----
            if self.augment is not None:
                rdct, ldct = self.augment.aug(rdct, ldct)

            if self.discriminator is not None and self.optimizer_d is not None:
                # ======== GAN training ========
                self._train_gan_step(ldct, rdct, loss_meters, pbar, step, epoch)
            else:
                # ======== Standard training ========
                self._train_standard_step(
                    ldct, rdct, loss_meters, pbar, step, epoch,
                    nf=data.get("nf"), meas=data.get("meas"))

            batch_time.update(time.time() - end)
            end = time.time()

        # Epoch-end logging：每个损失分项都记一条标量（mse/focal/charbonnier/...）
        for name, meter in loss_meters.items():
            self.writer.add_scalar(f"loss/train_{name}", meter.avg, epoch + 1)
        self.scheduler.step()
        if self.scheduler_d is not None:
            self.scheduler_d.step()

        if self.verbose:
            parts = [f"{name.upper()} {meter.avg:.6f}"
                     for name, meter in loss_meters.items()]
            print(f"Train Epoch {epoch + 1}: " + "  ".join(parts))

    @staticmethod
    def _update_loss_meters(loss_meters, loss_dict, n):
        """把 loss_dict 里的每个分项累积进对应的 AverageMeter。"""
        for name, val in loss_dict.items():
            if name not in loss_meters:
                loss_meters[name] = AverageMeter()
            loss_meters[name].update(val.item(), n)

    def _train_standard_step(self, ldct, rdct, loss_meters, pbar, step, epoch,
                             nf=None, meas=None):
        cfg = self.cfg
        model = self.model

        model_out = model(ldct)
        predict_for_viz = model_out

        if cfg.model.artifact_output and isinstance(model_out, tuple):
            # 伪影残差输出：(clean, artifact)；真实伪影 = 输入 - 干净GT
            pred_clean, pred_artifact = model_out
            gt_artifact = ldct - rdct
            loss, loss_dict = self._compute_loss(
                pred_clean, rdct, pred_artifact=pred_artifact, gt_artifact=gt_artifact)
            predict_for_viz = pred_clean

        elif cfg.loss.deep_supervision and isinstance(model_out, (tuple, list)):
            # Deep supervision: multiple outputs
            # model_out = (conv5, up4, up3, up2, final)
            final_out = model_out[-1]
            loss, loss_dict = self._compute_loss(final_out, rdct)

            # Add multi-scale losses from intermediate outputs
            lc = cfg.loss
            for idx, out in enumerate(model_out[:-1]):
                # Resize target to match intermediate output size
                target_resized = nn.functional.interpolate(
                    rdct, size=out.shape[2:], mode="bilinear", align_corners=False
                )
                if lc.mse_weight > 0:
                    loss += lc.mse_weight * self.criterion_mse(out, target_resized) * 0.1
            predict_for_viz = final_out

        else:
            loss, loss_dict = self._compute_loss(model_out, rdct)

        # ---- 物理一致性损失（可选）----
        if self.phys is not None:
            l_phys, rho = self._compute_phys_loss(
                predict_for_viz, rdct, nf, meas, epoch)
            if l_phys is not None:
                pc = cfg.physics
                w = pc.loss_weight * min(
                    1.0,
                    (epoch - pc.start_epoch + 1) / max(pc.warmup_epochs, 1))
                loss = loss + w * l_phys
                loss_dict["l_phys"] = l_phys.detach()
                loss_dict["rho"] = rho.detach()
                loss_dict["lambda_eff"] = torch.tensor(
                    w, device=ldct.device, dtype=ldct.dtype)

        self.optimizer.zero_grad()
        loss.backward()
        if self.phys is not None and cfg.physics.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.physics.grad_clip)
        self.optimizer.step()

        self._update_loss_meters(loss_meters, loss_dict, ldct.size(0))
        self.global_step += 1

        if step % cfg.training.log_interval == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        # Image logging at first step of epoch
        if step == 0 and (cfg.training.viz_interval > 0):
            self._log_images("train", ldct, rdct, predict_for_viz, epoch)

    def _train_gan_step(self, ldct, rdct, loss_meters, pbar, step, epoch):
        from src.models.discriminator import compute_gradient_penalty
        from src.utils import extract_patches_online
        cfg = self.cfg

        model = self.model
        discriminator = self.discriminator

        # --- Train Discriminator ---
        self.optimizer_d.zero_grad()
        with torch.no_grad():
            fake_imgs = model(ldct)

        # Extract patches for discriminator
        real_patches = extract_patches_online(rdct, 4)
        fake_patches = extract_patches_online(fake_imgs, 4)

        real_pred = discriminator(real_patches)
        fake_pred = discriminator(fake_patches)
        gp = compute_gradient_penalty(discriminator, real_patches, fake_patches)

        d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + cfg.gan.gp_weight * gp
        d_loss.backward()
        self.optimizer_d.step()

        # --- Train Generator ---
        self.optimizer.zero_grad()
        predict = model(ldct)
        fake_patches = extract_patches_online(predict, 4)
        fake_pred = discriminator(fake_patches)

        recon_loss, loss_dict = self._compute_loss(predict, rdct)
        adv_loss = -torch.mean(fake_pred)
        g_loss = recon_loss + cfg.loss.adv_weight * adv_loss

        g_loss.backward()
        self.optimizer.step()

        self._update_loss_meters(loss_meters, loss_dict, ldct.size(0))
        self.global_step += 1

        if step % cfg.training.log_interval == 0:
            pbar.set_postfix({"G": f"{g_loss.item():.4f}", "D": f"{d_loss.item():.4f}"})

        if step == 0:
            self._log_images("train", ldct, rdct, predict, epoch)

    @torch.no_grad()
    def _validate(self, epoch):
        model = self.model
        model.eval()

        loss_mse_scalar = AverageMeter()
        loss_ssim_scalar = AverageMeter()
        loss_psnr_scalar = AverageMeter()

        pbar = tqdm(self.valid_loader, desc=f"Valid Epoch {epoch + 1}",
                     leave=False, disable=not self.verbose)

        for data in pbar:
            ldct = data["ldct"].to(self.device, non_blocking=True)
            rdct = data["rdct"].to(self.device, non_blocking=True)

            predict = model(ldct)
            if self.cfg.model.artifact_output and isinstance(predict, (tuple, list)):
                predict = predict[0]      # artifact_output 时第一个输出是 clean
            elif isinstance(predict, (tuple, list)):
                predict = predict[-1]     # deep supervision 时最后一个输出

            l_mse = self.criterion_mse(predict, rdct)
            l_ssim = self.criterion_ssim(predict, rdct)
            psnr_val = -10 * math.log10(l_mse.item() + 1e-8)

            loss_mse_scalar.update(l_mse.item(), ldct.size(0))
            loss_ssim_scalar.update(l_ssim.item(), ldct.size(0))
            loss_psnr_scalar.update(psnr_val, ldct.size(0))

            pbar.set_postfix({"MSE": f"{l_mse.item():.6f}"})

        self.writer.add_scalar("loss/valid_mse", loss_mse_scalar.avg, epoch + 1)
        self.writer.add_scalar("loss/valid_ssim", loss_ssim_scalar.avg, epoch + 1)
        self.writer.add_scalar("valid/psnr", loss_psnr_scalar.avg, epoch + 1)
        self._log_images("valid", ldct, rdct, predict, epoch)

        if self.verbose:
            print(f"Valid Epoch {epoch + 1}: MSE {loss_mse_scalar.avg:.6f}  "
                  f"PSNR {loss_psnr_scalar.avg:.2f}  "
                  f"SSIM {loss_ssim_scalar.avg:.4f}")

    def _log_images(self, tag, ldct, rdct, predict, epoch):
        """Log image summaries to TensorBoard."""
        self.writer.add_image(f"{tag} img/ldct", normalization(ldct[0, :1]), epoch + 1)
        self.writer.add_image(f"{tag} img/reference", normalization(rdct[0, :1]), epoch + 1)
        self.writer.add_image(f"{tag} img/predict", normalization(predict[0]), epoch + 1)
        residual = torch.abs(rdct[0] - predict[0])
        self.writer.add_image(f"{tag} img/residual", normalization(residual), epoch + 1)

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def _try_resume(self):
        """Try to resume from the latest checkpoint in save_dir."""
        if not os.path.isdir(self.save_dir):
            return False

        try:
            checkpoint_path = find_lastest_file(self.save_dir)
        except (IndexError, FileNotFoundError):
            return False

        if self.verbose:
            print(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        load_model_model_parallel(self.model, checkpoint)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # Extract epoch from filename
        import re
        match = re.search(r"epoch_(\d+)", checkpoint_path)
        if match:
            self.current_epoch = int(match.group(1))

        return True

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self):
        """Run the full training loop."""
        if self.cfg.training.resume:
            resumed = self._try_resume()
            if not resumed and self.verbose:
                print("No checkpoint found, starting from scratch.")

        print("=" * 20 + " Start Training " + "=" * 20)

        for epoch in range(self.current_epoch, self.cfg.training.epochs):
            self.current_epoch = epoch
            print("=" * 20 + f" Epoch {epoch + 1:03d} " + "=" * 20)

            self._train_one_epoch(epoch)
            self._validate(epoch)

            # Save checkpoint
            if (epoch + 1) % self.cfg.training.save_interval == 0:
                save_model(self.model, self.optimizer, self.scheduler,
                           epoch + 1, self.save_dir)

        self.writer.close()
        if self.verbose:
            print("Training complete.")
