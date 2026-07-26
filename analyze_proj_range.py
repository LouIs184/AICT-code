#!/usr/bin/env python3
"""
Analyze projection data range across all patients.

Used to determine the optimal norm_clip_max value for proj_reader.

Usage:
    python analyze_proj_range.py --data-dir /path/to/UIH_data
"""

import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description="Analyze projection data range.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing patient subfolders.")
    args = parser.parse_args()

    patients = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ])
    print(f"Found {len(patients)} patients\n")

    global_min = float("inf")
    global_max = float("-inf")

    for p in patients:
        fpath = os.path.join(
            args.data_dir, p, "proj_pbi_fs_dec_blur_phase_100000.raw"
        )
        if not os.path.exists(fpath):
            print(f"  [skip] {p} — file not found")
            continue

        data = np.fromfile(fpath, dtype=np.float32).reshape(720, 128, 512)

        p_min = data.min()
        p_max = data.max()
        p_mean = data.mean()
        p_std = data.std()
        p_p99_9 = np.percentile(data, 99.9)
        neg_ratio = (data < 0).sum() / data.size * 100

        global_min = min(global_min, p_min)
        global_max = max(global_max, p_max)

        print(f"  {p}")
        print(f"    Range: [{p_min:.6f}, {p_max:.6f}]  "
              f"Mean: {p_mean:.6f}  Std: {p_std:.6f}")
        print(f"    99.9%: {p_p99_9:.6f}  Negatives: {neg_ratio:.4f}%\n")

    print("=" * 60)
    print("Global statistics:")
    print(f"  Global min: {global_min:.6f}")
    print(f"  Global max: {global_max:.6f}")
    suggested_clip = np.ceil(global_max * 10000) / 10000
    print(f"  Suggested norm_clip_max: {suggested_clip:.4f}")
    print(f"  Suggested clip range: [0, {suggested_clip:.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
