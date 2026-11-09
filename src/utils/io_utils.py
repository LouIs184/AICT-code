"""
Model checkpoint save / load utilities, plus the patient split loader.

Consolidates the original load_model / save_model variants from the AICT project.
"""

import os
import re
import torch


def _load_state_dict(net, state_dict):
    """Load weights into ``net`` (or ``net.module`` when DataParallel-wrapped).

    ``save_model`` strips the ``module.`` prefix, but checkpoints saved directly
    from a DataParallel model may still carry it; tolerate both forms.
    """
    target = net.module if hasattr(net, "module") else net
    if state_dict and any(k.startswith("module.") for k in state_dict):
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    target.load_state_dict(state_dict)


def load_model(net, checkpoint):
    """Load model state dict (single-GPU)."""
    _load_state_dict(net, checkpoint["state_dict"])
    return net


def load_model_data_parallel(net, checkpoint):
    """Load model and wrap in DataParallel."""
    _load_state_dict(net, checkpoint["state_dict"])
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        raise RuntimeError("DataParallel requested but multi-GPU unavailable.")
    return net


def load_model_model_parallel(net, checkpoint):
    """Load a checkpoint into a single- or multi-GPU model."""
    _load_state_dict(net, checkpoint["state_dict"])
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


def find_lastest_file(file_dir, prefix="model_at_epoch_", suffix=".dat"):
    """Return the checkpoint with the largest epoch number in ``file_dir``.

    Only files named ``model_at_epoch_<epoch>.dat`` are considered. The old
    implementation sorted *all* files by mtime, which picked up ``config.yaml``
    (rewritten at every start) and made ``--resume`` fail.
    """
    if not os.path.isdir(file_dir):
        raise FileNotFoundError(f"checkpoint directory not found: {file_dir}")

    pattern = re.compile(
        re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$"
    )
    candidates = []
    for name in os.listdir(file_dir):
        m = pattern.match(name)
        if m is not None:
            candidates.append((int(m.group(1)), name))

    if not candidates:
        raise FileNotFoundError(
            f"no {prefix}<epoch>{suffix} checkpoint found in {file_dir}"
        )

    candidates.sort()
    return os.path.join(file_dir, candidates[-1][1])


def load_patients(patients_config_path):
    """Load the train/valid patient split from a YAML file.

    Returns ``(train_list, valid_list)``. This is the single source of truth for
    the train/test split — used by prepare_data.py / prepare_proj_data.py (to
    generate the .txt index files) and by test.py / infer.py.
    """
    import yaml

    if not os.path.exists(patients_config_path):
        raise FileNotFoundError(
            f"Patient config not found: {patients_config_path}\n"
            "Expected a YAML file with 'train' and 'valid' patient lists "
            "(see patients.yaml)."
        )
    with open(patients_config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    train = [p for p in data.get("train", []) if p]
    valid = [p for p in data.get("valid", []) if p]
    if not train and not valid:
        raise ValueError(
            f"No 'train'/'valid' patient lists found in {patients_config_path}"
        )
    return train, valid
