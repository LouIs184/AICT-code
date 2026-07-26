"""
Configuration system for AICTVer2.

Provides dataclass-based configuration loaded from YAML files,
replacing the hardcoded hyperparameters in the original 9 training scripts.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Top-level experiment metadata."""
    name: str = "experiment"
    seed: int = 1234
    description: str = ""


@dataclass
class DataConfig:
    """Dataset and DataLoader settings."""
    reader: str = "raw"                     # "raw" | "dicom" | "proj"
    train_txt: str = "./train_img.txt"
    valid_txt: str = "./valid_img.txt"
    batch_size: int = 12
    valid_batch_size: int = 1
    num_workers: int = 8
    norm_clip_max: float = 0.28             # used by proj_reader
    shuffle: bool = True


@dataclass
class ModelConfig:
    """Model architecture settings."""
    name: str = "DenseUNet2d"               # model class name
    in_chl: int = 1
    out_chl: int = 1
    model_chl: int = 32


@dataclass
class LossConfig:
    """Loss function weights and settings."""
    mse_weight: float = 1.0
    charbonnier_weight: float = 0.0
    ssim_weight: float = 0.0
    vgg_weight: float = 0.0
    adv_weight: float = 0.01               # GAN adversarial loss weight (for generator)
    tv_weight: float = 0.0
    deep_supervision: bool = False         # multi-scale supervision for UNet2dDeepSuperviser
    charbonnier_eps: float = 1e-3


@dataclass
class OptimizerConfig:
    """Optimizer settings."""
    name: str = "Adam"                     # "Adam" | "AdamW" | "SophiaG"
    lr: float = 1e-4
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    """Learning rate scheduler settings."""
    name: str = "StepLR"                   # "StepLR" | "MultiStepLR" | "CosineAnnealingLR"
    step_size: int = 2
    gamma: float = 0.95
    milestones: Optional[list] = None      # for MultiStepLR
    t_max: int = 200                       # for CosineAnnealingLR


@dataclass
class GANConfig:
    """GAN-specific settings (used when loss.adv_weight > 0)."""
    d_lr: float = 1e-4
    d_betas: list = field(default_factory=lambda: [0.5, 0.9])
    d_iter: int = 1                        # discriminator updates per generator update
    gp_weight: float = 10.0               # gradient penalty weight
    n_critic: int = 1                      # how many D updates per G update


@dataclass
class AugmentConfig:
    """Data augmentation settings."""
    name: str = ""                         # "" | "mixup" | "cutmix" | "cutmix_v2"
    mixup_alpha: float = 1.2


@dataclass
class TrainingConfig:
    """Training loop settings."""
    epochs: int = 200
    save_dir: str = "./runs/checkpoints"
    log_dir: str = "./runs/logs"
    resume: bool = False
    log_interval: int = 20                 # steps between console log
    viz_interval: int = 1                  # epochs between TensorBoard image logging
    save_interval: int = 1                 # epochs between checkpoint saves


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def _merge_dict_into_dataclass(dc, data: dict):
    """Recursively merge keys from *data* into a (possibly nested) dataclass *dc*."""
    for key, val in data.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(val, dict):
            _merge_dict_into_dataclass(current, val)
        else:
            setattr(dc, key, val)


def load_config(path: str) -> Config:
    """
    Load a YAML configuration file and return a :class:`Config` instance.

    Only keys that exist on the dataclasses are applied — unknown keys are
    silently ignored, so adding comments in YAML is safe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return Config()

    cfg = Config()
    _merge_dict_into_dataclass(cfg, raw)
    return cfg


def save_config(cfg: Config, path: str):
    """Serialize *cfg* to YAML."""
    d = asdict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)
