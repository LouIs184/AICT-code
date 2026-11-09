#!/usr/bin/env python3
"""Evaluate a trained AICT checkpoint on a per-view list.

The script reuses the config (and therefore the exact crop / normalization /
model switches) that were used for training, so the metrics are directly
comparable to the trainer's validation PSNR. Prefer this over the legacy
``test.py`` / ``infer.py``, which hardcode the old (720, 128, 512) data layout.

Example::

    python scripts/eval_checkpoint.py \
        --config experiments/unet_nf0128_residual.yaml \
        --checkpoint runs/UNet_NF0128_Residual/checkpoints/model_at_epoch_040.dat \
        --split valid \
        --out runs/UNet_NF0128_Residual_eval.csv

Reports:
  * global MSE / PSNR (PSNR from the global MSE)
  * mean per-view PSNR and SSIM (same SSIM class as training)
  * mean |pred - input| and mean |pred - GT|
  * per-dose breakdown
"""

import argparse
import math
import os
import re
import sys

import torch
from torch.utils.data import DataLoader

# Ensure the project root is on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data import proj_reader
from src.losses import SSIM
from src.models import (UNet2d, UNet2dDeepSuperviser, DenseUNet2d,
                        PhysConsistency)
from src.utils.io_utils import load_model

MODEL_MAP = {
    "UNet2d": UNet2d,
    "UNet2dDeepSuperviser": UNet2dDeepSuperviser,
    "DenseUNet2d": DenseUNet2d,
}


def _dose_key(path):
    m = re.search(r"dose_1e(\d+)", path)
    return f"1e{m.group(1)}" if m else "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["valid", "train"], default="valid")
    ap.add_argument("--data-txt", default=None,
                    help="Override the split list from the config")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="Optional per-view CSV")
    ap.add_argument("--physics", action="store_true",
                    help="Also compute rho against the propagation target "
                         "(requires data.phys_file / heads_json)")
    ap.add_argument("--phys-file", default=None,
                    help="Override data.phys_file for --physics")
    ap.add_argument("--heads-json", default=None,
                    help="Override data.heads_json for --physics")
    ap.add_argument("--clip-meas", action="store_true",
                    help="Clip the physics target to [0, clip_max] "
                         "(legacy Restormer behaviour); default preserves signs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dc, mc = cfg.data, cfg.model

    phys_file = args.phys_file if args.phys_file is not None else dc.phys_file
    heads_json = (args.heads_json if args.heads_json is not None
                  else dc.heads_json)
    if args.physics and not phys_file:
        sys.exit("--physics requires data.phys_file (or --phys-file)")

    txt = args.data_txt or (dc.train_txt if args.split == "train" else dc.valid_txt)
    ds = proj_reader(
        paired_img_txt=txt,
        norm_clip_max=dc.norm_clip_max,
        crop_rows=dc.crop_rows,
        crop_cols=dc.crop_cols,
        view_shape=tuple(dc.view_shape),
        phys_file=phys_file if args.physics else "",
        heads_json=heads_json if args.physics else "",
        clip_meas=(dc.clip_meas or args.clip_meas),
    )
    if args.limit:
        ds.paired_files = ds.paired_files[: args.limit]

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_MAP.get(mc.name)
    if model_cls is None:
        sys.exit(f"unknown model: {mc.name} (available: {list(MODEL_MAP)})")
    model = model_cls(in_chl=mc.in_chl, out_chl=mc.out_chl, model_chl=mc.model_chl,
                      artifact_output=mc.artifact_output,
                      residual_output=mc.residual_output,
                      output_activation=mc.output_activation,
                      residual_init=mc.residual_init)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = load_model(model, ckpt)
    model = model.to(device).eval()

    criterion_ssim = SSIM(data_range=1.0).to(device)

    phys_mod = None
    if args.physics:
        phys_mod = PhysConsistency(float(dc.norm_clip_max),
                                   cfg.physics.pad).to(device)

    mse_sum = 0.0
    n_pix = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    ssim_n = 0
    corr_sum = 0.0
    err_sum = 0.0
    rho_sum = 0.0
    rho_n = 0
    n_views = 0
    per_dose = {}
    csv_rows = []

    for i, batch in enumerate(loader):
        ldct = batch["ldct"].to(device, non_blocking=True)
        rdct = batch["rdct"].to(device, non_blocking=True)
        meas_all = batch.get("meas")
        if meas_all is not None:
            meas_all = meas_all.to(device, non_blocking=True)
        nf_all = batch.get("nf")
        if nf_all is not None:
            nf_all = nf_all.to(device).reshape(-1)

        with torch.no_grad():
            pred = model(ldct)
            if isinstance(pred, (tuple, list)):
                pred = pred[0] if mc.artifact_output else pred[-1]

        b = ldct.size(0)
        # Compute metrics per view, not per batch: batch-level PSNR/SSIM would
        # be repeated for each view and could mix doses within one batch.
        for j in range(b):
            pred_j = pred[j:j + 1]
            gt_j = rdct[j:j + 1]
            in_j = ldct[j:j + 1]

            se_j = (pred_j - gt_j) ** 2
            mse_j = se_j.mean().item()
            psnr_j = -10.0 * math.log10(mse_j + 1e-8)
            ssim_j = criterion_ssim(pred_j, gt_j).item()
            corr_j = (pred_j - in_j).abs().mean().item()
            err_j = (pred_j - gt_j).abs().mean().item()

            rho_j = float("nan")
            if phys_mod is not None:
                if meas_all is None:
                    sys.exit("--physics but the dataset did not return 'meas' "
                             "(check data.phys_file)")
                if nf_all is not None:
                    nf_j = nf_all[j:j + 1]
                else:
                    nf_j = torch.full((1,), float(cfg.physics.nf),
                                      device=device)
                with torch.no_grad():
                    a_pred = phys_mod(pred_j, nf_j)
                    m_j = meas_all[j:j + 1]
                    sig = (m_j - gt_j).pow(2).flatten(1).mean(
                        1).sqrt().clamp_min(1e-6)
                    rho_j = ((a_pred - m_j).pow(2).flatten(1).mean(
                        1).sqrt() / sig)
                    if cfg.physics.rho_clip > 0:
                        rho_j = rho_j.clamp(max=cfg.physics.rho_clip)
                    rho_j = rho_j.mean().item()
                rho_sum += rho_j
                rho_n += 1

            mse_sum += se_j.sum().item()
            n_pix += se_j.numel()
            psnr_sum += psnr_j
            ssim_sum += ssim_j
            ssim_n += 1
            corr_sum += corr_j
            err_sum += err_j
            n_views += 1

            # DataLoader is not shuffled, so view i maps to line i.
            view_index = i * args.batch_size + j
            if view_index >= len(ds.paired_files):
                continue
            line = ds.paired_files[view_index].strip().split()
            path = line[0]
            dose = _dose_key(path)
            d = per_dose.setdefault(dose, {"n": 0, "psnr": 0.0, "ssim": 0.0,
                                           "corr": 0.0, "err": 0.0,
                                           "rho": 0.0, "rho_n": 0})
            d["n"] += 1
            d["psnr"] += psnr_j
            d["ssim"] += ssim_j
            d["corr"] += corr_j
            d["err"] += err_j
            if not math.isnan(rho_j):
                d["rho"] += rho_j
                d["rho_n"] += 1
            csv_rows.append((path, line[2] if len(line) > 2 else "",
                             dose, mse_j, psnr_j, ssim_j, corr_j, err_j,
                             rho_j))

        if n_views % 500 < b:
            print(f"  {n_views}/{len(ds)} views", end="\r", flush=True)

    print()
    if n_views == 0:
        sys.exit("no views evaluated")
    global_mse = mse_sum / n_pix
    global_psnr = -10.0 * math.log10(global_mse + 1e-8)

    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"split     : {args.split}  ({n_views} views)")
    print(f"global MSE : {global_mse:.6e}")
    print(f"global PSNR: {global_psnr:.4f} dB")
    print(f"mean PSNR  : {psnr_sum / n_views:.4f} dB")
    print(f"mean SSIM  : {ssim_sum / max(ssim_n, 1):.4f}")
    print(f"mean |pred-input|: {corr_sum / n_views:.6e}")
    print(f"mean |pred-GT|   : {err_sum / n_views:.6e}")
    if rho_n:
        print(f"mean rho   : {rho_sum / rho_n:.4f}")
    elif args.physics:
        print("mean rho   : n/a (no valid propagation target)")

    print("\nper dose:")
    print(f"{'dose':>8}  {'views':>6}  {'PSNR(dB)':>9}  {'SSIM':>7}  "
          f"{'rho':>7}  {'|pred-input|':>13}  {'|pred-GT|':>12}")
    for dose in sorted(per_dose):
        d = per_dose[dose]
        n = d["n"]
        rho_str = f"{d['rho'] / d['rho_n']:.4f}" if d["rho_n"] else "n/a"
        print(f"{dose:>8}  {n:>6}  {d['psnr'] / n:>9.4f}  {d['ssim'] / n:>7.4f}  "
              f"{rho_str:>7}  {d['corr'] / n:>13.6e}  {d['err'] / n:>12.6e}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("input,view,dose,mse,psnr,ssim,mean_abs_pred_input,"
                    "mean_abs_pred_gt,rho\n")
            for r in csv_rows:
                f.write(",".join(str(v) for v in r) + "\n")
        print(f"\nper-view CSV: {args.out}")


if __name__ == "__main__":
    main()
