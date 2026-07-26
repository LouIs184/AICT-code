"""
AICTVer2 — AI CT Image Reconstruction, refactored.

Following the architecture analysis in AICT/docs/ARCHITECTURE_ANALYSIS.md,
this package restructures the original AICT codebase into a maintainable,
modular layout with a unified training engine and YAML-based configuration.
"""

from . import config
from . import models
from . import losses
from . import data
from . import utils
from . import optimizers
from .trainer import Trainer
from .config import Config, load_config

__all__ = [
    "config",
    "models",
    "losses",
    "data",
    "utils",
    "optimizers",
    "Trainer",
    "Config",
    "load_config",
]
