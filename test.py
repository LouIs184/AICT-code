#!/usr/bin/env python3
"""
AICTVer2 inference / testing entry point.

Supports both PhaseCT (image domain) and ProjCT (projection domain) inference.

Usage:
    python test.py --config experiments/denseunet_phasect.yaml \\
                   --checkpoint ./runs/DenseUNet_PhaseCT/checkpoints/model_at_epoch_200.dat \\
                   --data-dir /path/to/data \\
                   --output ./results/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.models import DenseUNet2d, UNet2d
from src.utils.io_utils import load_model

MODEL_MAP = {
    "DenseUNet2d": DenseUNet2d,
    "UNet2d": UNet2d,
}


def main():
    parser = argparse.ArgumentParser(description="AICTVer2 — Run inference.")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to experiment config (for model params).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.dat).")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing patient data.")
    parser.add_argument("--output", "-o", type=str, default="./results/",
                        help="Output directory for predictions.")
    parser.add_argument("--patient", type=str, nargs="+", default=None,
                        help="Patient folder name(s). If omitted, use defaults.")
    parser.add_argument("--domain", type=str, choices=["phase", "proj"], default="phase",
                        help="Domain: 'phase' (image) or 'proj' (projection).")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID.")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.output, exist_ok=True)

    # Load config for model parameters
    cfg = load_config(args.config)
    mc = cfg.model

    # Build model
    model_cls = MODEL_MAP.get(mc.name, DenseUNet2d)
    model = model_cls(in_chl=mc.in_chl, out_chl=mc.out_chl, model_chl=mc.model_chl)
    model = load_model(model, torch.load(args.checkpoint, map_location="cpu"))
    model = model.cuda()
    model.eval()

    if args.domain == "phase":
        _infer_phase(args, model)
    else:
        _infer_proj(args, model)


def _infer_phase(args, model):
    """Image-domain inference (PhaseCT): process (160, 512, 512) volumes."""
    default_patients = [
        "17238_33314_960+_AXIAL_CE1_F063Y_20220106_Thick1_Incre1",
        "38423_35948_960+_AXIAL_CE1_M049Y_20200601_Thick1_Incre1",
    ]
    patients = args.patient or default_patients

    for patient in patients:
        print(f"\nTesting {patient} ...")
        patient_dir = os.path.join(args.data_dir, patient)

        vol = np.fromfile(
            os.path.join(patient_dir, "rec_pbi_fs_dec_blur_phase_100000.raw"),
            dtype=np.float32,
        ).reshape(160, 512, 512)

        gt = np.fromfile(
            os.path.join(patient_dir, "rec_no_pbi.raw"),
            dtype=np.float32,
        ).reshape(160, 512, 512)

        preds = []
        for i in tqdm(range(1, 159), desc=f"  Inferring {patient}"):
            s = ((vol[i] + 1000) / 2000).astype(np.float32)
            s = np.clip(s, 0, 1)
            t = torch.FloatTensor(s[np.newaxis, np.newaxis, ...]).cuda()
            with torch.no_grad():
                p = model(t)
            p = np.squeeze(p.cpu().numpy()) * 2000 - 1000  # denormalize
            preds.append(p)

        pred_vol = np.array(preds)  # (158, 512, 512)
        pred_vol.tofile(os.path.join(args.output, f"{patient}_pred.raw"))
        print(f"  Saved to {args.output}/{patient}_pred.raw")

        # Compute metrics
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn

        gt_slices = gt[1:-1]
        psnr_val = psnr_fn(gt_slices, pred_vol,
                           data_range=gt_slices.max() - gt_slices.min())
        ssim_val = np.mean([
            ssim_fn(gt_slices[i], pred_vol[i],
                    data_range=gt_slices[i].max() - gt_slices[i].min())
            for i in range(158)
        ])
        print(f"  PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}")


def _infer_proj(args, model):
    """Projection-domain inference (ProjCT): process (720, 128, 512) volumes."""
    default_patients = [
        "17238_33314_960+_AXIAL_CE1_F063Y_20220106_Thick1_Incre1",
        "38423_35948_960+_AXIAL_CE1_M049Y_20200601_Thick1_Incre1",
    ]
    patients = args.patient or default_patients

    # Read crop_rows from config (same as training)
    from src.config import load_config
    infer_cfg = load_config(args.config)
    crop_rows = getattr(infer_cfg.data, "crop_rows", 0)
    target_rows = 128 - 2 * crop_rows if crop_rows > 0 else 128

    for patient in patients:
        print(f"\nTesting {patient} (proj domain) ...")
        patient_dir = os.path.join(args.data_dir, patient)

        vol = np.fromfile(
            os.path.join(patient_dir, "proj_pbi_fs_dec_blur_phase_100000.raw"),
            dtype=np.float32,
        ).reshape(720, 128, 512)

        gt = np.fromfile(
            os.path.join(patient_dir, "proj_no_pbi.raw"),
            dtype=np.float32,
        ).reshape(720, 128, 512)

        norm_clip_max = 0.28
        preds = []
        for i in tqdm(range(720), desc=f"  Inferring {patient}"):
            s = vol[i]  # (128, 512)
            # Crop black border rows (must match training)
            if crop_rows > 0:
                s = s[crop_rows:-crop_rows, :]
            s = (np.clip(s, 0, norm_clip_max) / norm_clip_max).astype(np.float32)
            t = torch.FloatTensor(s[np.newaxis, np.newaxis, ...]).cuda()
            with torch.no_grad():
                p = model(t)
            pred = np.squeeze(p.cpu().numpy()) * norm_clip_max
            preds.append(pred)

        pred_vol = np.array(preds)  # (720, target_rows, 512)
        pred_vol.tofile(os.path.join(args.output, f"{patient}_pred.raw"))
        print(f"  Saved to {args.output}/{patient}_pred.raw (shape {pred_vol.shape})")


if __name__ == "__main__":
    main()
