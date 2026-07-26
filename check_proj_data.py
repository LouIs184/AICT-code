#!/usr/bin/env python3
"""
Debug script to explore projection data shape and layout.

Tries multiple reshape strategies for the (720, 128, 512) projection volume
and saves a diagnostic image grid to identify the correct layout.

Usage:
    python check_proj_data.py --data-dir /path/to/UIH_data --patient "patient_folder"
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    parser = argparse.ArgumentParser(description="Explore projection data shape.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory of patient data.")
    parser.add_argument("--patient", type=str, required=True,
                        help="Patient folder name.")
    parser.add_argument("--output", "-o", type=str, default="./proj_reshape_debug.png")
    args = parser.parse_args()

    raw_path = os.path.join(
        args.data_dir, args.patient,
        "proj_pbi_fs_dec_blur_phase_100000.raw"
    )
    raw = np.fromfile(raw_path, dtype=np.float32)
    print(f"Total elements: {raw.shape[0]}")

    shapes = [
        ("(512, 128, 720)  = (ch, row, view)", (512, 128, 720)),
        ("(512, 720, 128)  = (ch, view, row)", (512, 720, 128)),
        ("(128, 512, 720)  = (row, ch, view)", (128, 512, 720)),
        ("(128, 720, 512)  = (row, view, ch)", (128, 720, 512)),
        ("(720, 512, 128)  = (view, ch, row)", (720, 512, 128)),
        ("(720, 128, 512)  = (view, row, ch)", (720, 128, 512)),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, (label, shape) in zip(axes.flat, shapes):
        try:
            data = raw.copy().reshape(shape)
            if shape[0] == 720:
                img = data[0]
            else:
                img = data[:, :, 0]

            im = ax.imshow(img, aspect="auto", cmap="gray")
            ax.set_title(label)
            plt.colorbar(im, ax=ax, fraction=0.046)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            ax.set_title(label)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved diagnostic image to {args.output}")


if __name__ == "__main__":
    main()
