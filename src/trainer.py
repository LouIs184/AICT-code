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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from src.config import Config
from src.models import UNet2d, UNet2dDeepSuperviser, DenseUNet2d, Discriminator2D
from src.losses import CharbonnierLoss, SSIM, vgg_feature_extractor, vgg_loss_calc, TVLoss
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cudnn.benchmark = True

        self._build_dataloaders()
        self._build_model()
        self._build_losses()
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
        if dc.reader == "proj":
            reader_kwargs["norm_clip_max"] = dc.norm_clip_max

        self.train_dataset = reader_cls(paired_img_txt=dc.train_txt, **reader_kwargs)
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

        model = model_cls(in_chl=mc.in_chl, out_chl=mc.out_chl, model_chl=mc.model_chl)

        # Multi-GPU
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        self.model = model

        # GAN discriminator
        self.discriminator = None
        if self.cfg.loss.adv_weight > 0:
            self.discriminator = Discriminator2D().to(self.device)

    def _build_losses(self):
        lc = self.cfg.loss
        self.criterion_mse = nn.MSELoss()
        self.criterion_charbonnier = CharbonnierLoss(eps=lc.charbonnier_eps)
        self.criterion_ssim = SSIM()
        self.criterion_tv = TVLoss()

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
        if os.path.exists(tc.save_dir):
            self.save_dir = tc.save_dir
        else:
            self.save_dir = os.path.join(tc.save_dir, self.cfg.experiment.name, "checkpoints")

        now = datetime.now().strftime("%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(tc.log_dir, self.cfg.experiment.name, now)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

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

    def _compute_loss(self, predict, target, model_out=None):
        """Compute combined loss from configured weights."""
        lc = self.cfg.loss
        loss = 0.0
        losses = {}

        if lc.mse_weight > 0:
            l_mse = self.criterion_mse(predict, target)
            losses["mse"] = l_mse
            loss += lc.mse_weight * l_mse

        if lc.charbonnier_weight > 0:
            l_cl = self.criterion_charbonnier(predict, target)
            losses["charbonnier"] = l_cl
            loss += lc.charbonnier_weight * l_cl

        if lc.ssim_weight > 0:
            l_ssim = 1 - self.criterion_ssim(predict, target)
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

        return loss, losses

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
        loss_mse_scalar = AverageMeter()
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
                self._train_gan_step(ldct, rdct, loss_mse_scalar, pbar, step, epoch)
            else:
                # ======== Standard training ========
                self._train_standard_step(ldct, rdct, loss_mse_scalar, pbar, step, epoch)

            batch_time.update(time.time() - end)
            end = time.time()

        # Epoch-end logging
        self.writer.add_scalar("loss/train_mse", loss_mse_scalar.avg, epoch + 1)
        self.scheduler.step()
        if self.scheduler_d is not None:
            self.scheduler_d.step()

        if self.verbose:
            print(f"Train Epoch {epoch + 1}: MSE {loss_mse_scalar.avg:.6f}")

    def _train_standard_step(self, ldct, rdct, loss_meter, pbar, step, epoch):
        cfg = self.cfg
        model = self.model

        predict = model(ldct)

        if cfg.loss.deep_supervision and isinstance(predict, (tuple, list)):
            # Deep supervision: multiple outputs
            # predict = (conv5, up4, up3, up2, final)
            final_out = predict[-1]
            loss, loss_dict = self._compute_loss(final_out, rdct)

            # Add multi-scale losses from intermediate outputs
            lc = cfg.loss
            for idx, out in enumerate(predict[:-1]):
                # Resize target to match intermediate output size
                target_resized = nn.functional.interpolate(
                    rdct, size=out.shape[2:], mode="bilinear", align_corners=False
                )
                if lc.mse_weight > 0:
                    loss += lc.mse_weight * self.criterion_mse(out, target_resized) * 0.1
        else:
            loss, loss_dict = self._compute_loss(predict, rdct)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_meter.update(loss_dict.get("mse", loss).item(), ldct.size(0))
        self.global_step += 1

        if step % cfg.training.log_interval == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        # Image logging at first step of epoch
        if step == 0 and (cfg.training.viz_interval > 0):
            self._log_images("train", ldct, rdct, predict[-1] if isinstance(predict, tuple) else predict, epoch)

    def _train_gan_step(self, ldct, rdct, loss_meter, pbar, step, epoch):
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

        loss_meter.update(loss_dict.get("mse", g_loss).item(), ldct.size(0))
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
            if isinstance(predict, (tuple, list)):
                predict = predict[-1]

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
