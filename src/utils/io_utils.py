"""
Model checkpoint save / load utilities.

Consolidates the original load_model / save_model variants from the AICT project.
"""

import os
import torch


def load_model(net, checkpoint):
    """Load model state dict (single-GPU)."""
    net.load_state_dict(checkpoint["state_dict"])
    return net


def load_model_data_parallel(net, checkpoint):
    """Load model and wrap in DataParallel."""
    net.load_state_dict(checkpoint["state_dict"])
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        raise RuntimeError("DataParallel requested but multi-GPU unavailable.")
    return net


def load_model_model_parallel(net, checkpoint):
    """Load model (works with both single and multi-GPU checkpoints)."""
    net.load_state_dict(checkpoint["state_dict"])
    return net


def save_model(net, optimizer, scheduler, epoch, save_dir):
    """Save model, optimizer, and scheduler state to a checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)

    state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()

    path = os.path.join(save_dir, f"model_at_epoch_{epoch:03d}.dat")
    torch.save({
        "state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")
    return path


def find_lastest_file(file_dir):
    """Find the most recently modified file in a directory."""
    files = os.listdir(file_dir)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(file_dir, x)))
    return os.path.join(file_dir, files[-1])
