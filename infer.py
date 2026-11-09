#!/usr/bin/env python3
"""
Flexible inference script — run on any patient (train, valid, or custom).

Preserves the exact same inference logic, metrics computation, and file output
structure as test.py, but allows selecting patients by group or by name.

Usage:
    # All validation patients (same as test.py defaults)
    python infer.py --patients-config patients.yaml \\
        --config experiments/denseunet_phasect.yaml \\
        --checkpoint ./runs/DenseUNet_PhaseCT/checkpoints/model_at_epoch_200.dat \\
        --data-dir /path/to/data \\
        --domain phase --patient valid

    # All training patients
    python infer.py ... --patient train

    # All patients (train + valid)
    python infer.py ... --patient all

    # Specific patients
    python infer.py ... --patient "patient_A" "patient_B"

    # Projection domain
    python infer.py ... --domain proj --patient all
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _preset_cuda_visible_devices(argv):
    """Set CUDA_VISIBLE_DEVICES before torch is imported.

    Setting it after ``import torch`` leaves PyTorch with a stale device list
    and raises ``device >= 0 && device < num_gpus`` on ``.cuda()``.
    """
    for i, arg in enumerate(argv):
        if arg == "--gpu" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]
        elif arg.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]


_preset_cuda_visible_devices(sys.argv)

import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.models import DenseUNet2d, UNet2d

MODEL_MAP = {
    "DenseUNet2d": DenseUNet2d,
    "UNet2d": UNet2d,
}


def load_patients(patients_config_path):
    """Load patient list YAML, return dict with 'train' and 'valid' keys."""
    from src.utils.io_utils import load_patients as _load_patients

    train, valid = _load_patients(patients_config_path)
    return {"train": train, "valid": valid}


def resolve_patients(patient_config, patient_arg):
    """Resolve the --patient argument into a flat list of patient names."""
    if patient_arg is None or "all" in patient_arg:
        return patient_config["train"] + patient_config["valid"]
    if "train" in patient_arg and "valid" in patient_arg:
        return patient_config["train"] + patient_config["valid"]
    if "train" in patient_arg:
        return patient_config["train"]
    if "valid" in patient_arg:
        return patient_config["valid"]
    # Specific patient name(s)
    return patient_arg


def _compute_metrics(pred, gt, tag=""):
    """Compute MSE, RMSE, MAE, PSNR, SSIM between pred and gt."""
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - gt))
    psnr_val = psnr_fn(gt, pred, data_range=gt.max() - gt.min())
    ssim_val = np.mean([
        ssim_fn(gt[i], pred[i], data_range=gt[i].max() - gt[i].min())
        for i in range(gt.shape[0])
    ])

    label = f"  {tag}" if tag else ""
    print(f"{label}  MSE: {mse:.6f}  RMSE: {rmse:.6f}  MAE: {mae:.6f}  "
          f"PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}")
    return {"mse": mse, "rmse": rmse, "mae": mae, "psnr": psnr_val, "ssim": ssim_val}


def infer_phase(patient, model, args):
    """Image-domain inference for one patient."""
    cfg = load_config(args.config)
    hu_min = getattr(cfg.data, "hu_min", -1000.0)
    hu_range = getattr(cfg.data, "hu_max", 1000.0) - hu_min

    patient_dir = os.path.join(args.data_dir, patient)

    vol = np.fromfile(
        os.path.join(patient_dir, "rec_pbi_fs_dec_blur_phase_100000.raw"),
        dtype=np.float32,
    ).reshape(160, 512, 512)

    gt = np.fromfile(
        os.path.join(patient_dir, "rec_no_pbi.raw"),
        dtype=np.float32,
    ).reshape(160, 512, 512)

    out_dir = os.path.join(args.output, "phase", patient)
    os.makedirs(out_dir, exist_ok=True)

    preds = []
    for i in tqdm(range(1, 159), desc=f"  Inferring {patient}"):
        s = ((vol[i] - hu_min) / hu_range).astype(np.float32)
        s = np.clip(s, 0, 1)
        t = torch.FloatTensor(s[np.newaxis, np.newaxis, ...]).cuda()
        with torch.no_grad():
            p = model(t)
        if isinstance(p, (tuple, list)):
            p = p[0]   # artifact_output 模型返回 (clean, artifact)，取 clean
        p = np.squeeze(p.cpu().numpy()) * hu_range + hu_min
        preds.append(p)

    pred_vol = np.array(preds)  # (158, 512, 512)
    gt_slices = gt[1:-1]        # (158, 512, 512)
    input_slices = vol[1:-1]    # (158, 512, 512)

    pred_vol.tofile(os.path.join(out_dir, "pred.raw"))
    gt_slices.tofile(os.path.join(out_dir, "gt.raw"))
    input_slices.tofile(os.path.join(out_dir, "input.raw"))
    print(f"  Saved input / pred / gt to {out_dir}")

    print(f"  ── Metrics for {patient} ──")
    _compute_metrics(input_slices, gt_slices, tag="input vs gt :")
    _compute_metrics(pred_vol, gt_slices, tag="pred  vs gt :")


def _first_existing(dir_path, names):
    """Return the first existing filename in ``names`` (for input-file fallback)."""
    for name in names:
        if os.path.exists(os.path.join(dir_path, name)):
            return name
    raise FileNotFoundError(f"none of {names} found in {dir_path}")


def infer_proj(patient, model, args):
    """Projection-domain inference with config-driven shape / crop / columns."""
    from src.config import load_config as _load_config
    from src.data.proj_reader import crop_projection, read_raw_volume

    infer_cfg = _load_config(args.config)
    crop_rows = getattr(infer_cfg.data, "crop_rows", 0)
    crop_cols = getattr(infer_cfg.data, "crop_cols", 0)
    norm_clip_max = getattr(infer_cfg.data, "norm_clip_max", 0.28)
    view_shape = tuple(getattr(infer_cfg.data, "view_shape", [128, 512]))

    patient_dir = os.path.join(args.data_dir, patient)
    input_name = args.input_name or _first_existing(patient_dir, [
        "proj_pbi_fs_dec_blur_phase_100000.raw",
        "proj_pbi_fs_dec_blur_phase.raw",
    ])
    gt_name = args.gt_name or "proj_no_pbi.raw"

    vol = read_raw_volume(os.path.join(patient_dir, input_name), view_shape)
    gt = read_raw_volume(os.path.join(patient_dir, gt_name), view_shape)
    n_views = vol.shape[0]
    print(f"  input: {input_name}  shape={vol.shape}")

    out_dir = os.path.join(args.output, "proj", patient)
    os.makedirs(out_dir, exist_ok=True)

    preds = []
    for i in tqdm(range(n_views), desc=f"  Inferring {patient}"):
        s = crop_projection(vol[i], crop_rows, crop_cols)
        s = (np.clip(s, 0, norm_clip_max) / norm_clip_max).astype(np.float32)
        t = torch.FloatTensor(s[np.newaxis, np.newaxis, ...]).cuda()
        with torch.no_grad():
            p = model(t)
        if isinstance(p, (tuple, list)):
            p = p[0]   # artifact_output 模型返回 (clean, artifact)，取 clean
        pred = np.squeeze(p.cpu().numpy()) * norm_clip_max
        preds.append(pred)

    pred_vol = np.array(preds)  # (n_views, H', W')

    gt_cropped = crop_projection(gt, crop_rows, crop_cols)
    input_cropped = crop_projection(vol, crop_rows, crop_cols)

    pred_vol.tofile(os.path.join(out_dir, "pred.raw"))
    gt_cropped.tofile(os.path.join(out_dir, "gt.raw"))
    input_cropped.tofile(os.path.join(out_dir, "input.raw"))
    print(f"  Saved input / pred / gt to {out_dir} (shape {pred_vol.shape})")

    print(f"  ── Metrics for {patient} ──")
    _compute_metrics(input_cropped, gt_cropped, tag="input vs gt :")
    _compute_metrics(pred_vol, gt_cropped, tag="pred  vs gt :")


def main():
    parser = argparse.ArgumentParser(
        description="AICTVer2 — Flexible inference on any patient(s)."
    )
    parser.add_argument("--patients-config", type=str, required=True,
                        help="YAML file listing train/valid patients.")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Experiment YAML config (for model params).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.dat).")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing patient data.")
    parser.add_argument("--output", "-o", type=str, default="./results/",
                        help="Output directory for predictions.")
    parser.add_argument("--patient", type=str, nargs="+", default=["all"],
                        help="Patients to infer: 'train', 'valid', 'all', "
                             "or specific names (space-separated). Default: all")
    parser.add_argument("--domain", type=str, choices=["phase", "proj"],
                        default="phase",
                        help="Domain: 'phase' (image) or 'proj' (projection).")
    parser.add_argument("--input-name", type=str, default=None,
                        help="Input raw filename inside the patient dir. Default: "
                             "proj_pbi_fs_dec_blur_phase_100000.raw, falling back "
                             "to proj_pbi_fs_dec_blur_phase.raw.")
    parser.add_argument("--gt-name", type=str, default=None,
                        help="GT raw filename inside the patient dir "
                             "(default: proj_no_pbi.raw).")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID.")

    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES was already set from --gpu before importing torch.
    os.makedirs(args.output, exist_ok=True)

    # Load patient list
    patient_config = load_patients(args.patients_config)
    patients = resolve_patients(patient_config, args.patient)
    print(f"Resolved {len(patients)} patient(s) for inference.")

    # Build model
    cfg = load_config(args.config)
    mc = cfg.model
    model_cls = MODEL_MAP.get(mc.name, DenseUNet2d)
    model = model_cls(in_chl=mc.in_chl, out_chl=mc.out_chl, model_chl=mc.model_chl,
                      artifact_output=mc.artifact_output,
                      residual_output=mc.residual_output,
                      output_activation=mc.output_activation,
                      residual_init=mc.residual_init)
    model = load_model(model, torch.load(args.checkpoint, map_location="cpu"))
    model = model.cuda()
    model.eval()

    # Determine domain and run
    if args.domain == "phase":
        infer_fn = infer_phase
    else:
        infer_fn = infer_proj

    for patient in patients:
        print(f"\n{'='*60}")
        print(f"Patient: {patient}")
        print(f"{'='*60}")
        infer_fn(patient, model, args)


def load_model(net, checkpoint):
    """Load model state dict (single-GPU)."""
    from src.utils.io_utils import load_model as _load
    return _load(net, checkpoint)


if __name__ == "__main__":
    main()
