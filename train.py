#!/usr/bin/env python3
"""
AICTVer2 training entry point.

Usage:
    python train.py --config experiments/denseunet_phasect.yaml
    python train.py --config experiments/denseunet_projct.yaml --resume
"""

import argparse
import os
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="AICTVer2 — Train a CT reconstruction model.")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML configuration file.")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from the latest checkpoint.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override checkpoint save directory.")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Override log directory.")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Override CUDA_VISIBLE_DEVICES.")

    args = parser.parse_args()

    # GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load configuration
    cfg = load_config(args.config)

    # CLI overrides
    if args.resume:
        cfg.training.resume = True
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    if args.save_dir is not None:
        cfg.training.save_dir = args.save_dir
    if args.log_dir is not None:
        cfg.training.log_dir = args.log_dir

    # Create trainer and run
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
