#!/usr/bin/env python3
"""Measure the identity baseline (input vs GT) on an AICT per-view list.

This is the floor that any restoration model must beat. It uses exactly the
same crop / normalization as ``src/data/proj_reader.py`` so its PSNR is
directly comparable to the trainer's validation PSNR (normalized [0, 1]).

Example::

    python scripts/identity_baseline.py \
        --list nf0128_list/valid_nf0128_views.txt \
        --view_shape 500 500 --crop_rows 122 --crop_cols 2 --clip_max 0.047
"""

import argparse
import math
import os
import re
import sys

import numpy as np

try:
    from skimage.metrics import structural_similarity as sk_ssim
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SKIMAGE = False


def _read_view(path, view_idx, view_shape, count):
    view_size = int(np.prod(view_shape))
    byte_offset = view_idx * view_size * 4
    arr = np.fromfile(path, dtype=np.float32, count=view_size, offset=byte_offset)
    if arr.size != view_size:
        raise ValueError(
            f"short read from {path} at view {view_idx}: "
            f"got {arr.size} values, expected {view_size}"
        )
    return arr.reshape(view_shape)


def _crop(a, crop_rows, crop_cols):
    if crop_rows > 0:
        a = a[crop_rows:-crop_rows, :]
    if crop_cols > 0:
        a = a[:, crop_cols:-crop_cols]
    return a


def _dose_key(path):
    m = re.search(r"dose_1e(\d+)", path)
    return f"1e{m.group(1)}" if m else "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="Per-view list: input label view")
    ap.add_argument("--view_shape", type=int, nargs=2, default=[500, 500],
                    metavar=("H", "W"), help="Per-view shape (default 500 500)")
    ap.add_argument("--crop_rows", type=int, default=0)
    ap.add_argument("--crop_cols", type=int, default=0)
    ap.add_argument("--clip_max", type=float, default=0.047)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only evaluate the first N views (for a quick check)")
    ap.add_argument("--no_ssim", action="store_true", help="Skip SSIM")
    ap.add_argument("--out", default=None, help="Optional per-view CSV output")
    args = ap.parse_args()

    view_shape = tuple(args.view_shape)
    view_size = int(np.prod(view_shape))

    rows = []
    with open(args.list, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                sys.exit(f"bad line (need 3 columns): {line!r}")
            rows.append((parts[0], parts[1], int(parts[2])))
    if args.limit:
        rows = rows[: args.limit]

    # Validate file sizes once per unique path.
    checked = {}
    for inp, lab, _ in rows:
        for p in (inp, lab):
            if p in checked:
                continue
            size = os.path.getsize(p)
            if size % (view_size * 4) != 0:
                sys.exit(f"file size {size} not a multiple of {view_size * 4}: {p}")
            checked[p] = size // (view_size * 4)

    per_dose = {}
    csv_rows = []
    mse_sum = 0.0
    n_pix = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    ssim_n = 0
    mae_sum = 0.0
    corr_sum = 0.0

    for i, (inp, lab, view_idx) in enumerate(rows, 1):
        if view_idx >= checked[inp] or view_idx >= checked[lab]:
            sys.exit(f"view {view_idx} out of range for {inp} / {lab}")

        x = _read_view(inp, view_idx, view_shape, view_size)
        y = _read_view(lab, view_idx, view_shape, view_size)
        x = _crop(x, args.crop_rows, args.crop_cols)
        y = _crop(y, args.crop_rows, args.crop_cols)

        x = np.clip(x, 0, args.clip_max) / args.clip_max
        y = np.clip(y, 0, args.clip_max) / args.clip_max

        diff = x - y
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        corr = mae  # |input - GT| is the correction magnitude
        psnr = -10.0 * math.log10(mse + 1e-8)

        ssim = float("nan")
        if not args.no_ssim and _HAS_SKIMAGE:
            ssim = float(sk_ssim(y, x, data_range=1.0))

        dose = _dose_key(inp)
        d = per_dose.setdefault(dose, {"n": 0, "mse_sum": 0.0, "psnr_sum": 0.0,
                                       "ssim_sum": 0.0, "ssim_n": 0,
                                       "mae_sum": 0.0})
        d["n"] += 1
        d["mse_sum"] += mse
        d["psnr_sum"] += psnr
        d["mae_sum"] += mae
        if not math.isnan(ssim):
            d["ssim_sum"] += ssim
            d["ssim_n"] += 1

        mse_sum += mse * x.size
        n_pix += x.size
        psnr_sum += psnr
        mae_sum += mae
        corr_sum += corr
        if not math.isnan(ssim):
            ssim_sum += ssim
            ssim_n += 1
        csv_rows.append((inp, view_idx, dose, mse, psnr, mae, ssim))

        if i % 500 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} views", end="\r", flush=True)

    print()
    if not rows:
        sys.exit("no views to evaluate")

    global_mse = mse_sum / n_pix
    global_psnr = -10.0 * math.log10(global_mse + 1e-8)
    print(f"\nviews: {len(rows)}   crop: {view_shape} -> {x.shape}")
    print(f"global MSE : {global_mse:.6e}")
    print(f"global PSNR: {global_psnr:.4f} dB")
    print(f"mean PSNR  : {psnr_sum / len(rows):.4f} dB")
    print(f"mean MAE   : {mae_sum / len(rows):.6e}")
    print(f"mean |corr|: {corr_sum / len(rows):.6e}")
    if ssim_n:
        print(f"mean SSIM  : {ssim_sum / ssim_n:.4f}")

    print("\nper dose:")
    header = f"{'dose':>8}  {'views':>6}  {'PSNR(dB)':>9}  {'MAE':>11}  {'SSIM':>7}"
    print(header)
    for dose in sorted(per_dose):
        d = per_dose[dose]
        ssim_str = f"{d['ssim_sum'] / d['ssim_n']:.4f}" if d["ssim_n"] else "n/a"
        print(f"{dose:>8}  {d['n']:>6}  {d['psnr_sum'] / d['n']:>9.4f}  "
              f"{d['mae_sum'] / d['n']:>11.6e}  {ssim_str:>7}")

    if not _HAS_SKIMAGE and not args.no_ssim:
        print("\n[note] scikit-image not available; SSIM skipped. "
              "Install with `python -m pip install scikit-image`.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("input,view,dose,mse,psnr,mae,ssim\n")
            for r in csv_rows:
                f.write(",".join(str(v) for v in r) + "\n")
        print(f"\nper-view CSV: {args.out}")


if __name__ == "__main__":
    main()
