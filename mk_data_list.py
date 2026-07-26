#!/usr/bin/env python3
"""
Generate DICOM paired file list for AAPM datasets (legacy).

Usage:
    python mk_data_list.py --data-dir /path/to/AAPM --output ./
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Generate AAPM DICOM file pairs.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root of AAPM dataset.")
    parser.add_argument("--output", "-o", type=str, default="./",
                        help="Output directory.")
    args = parser.parse_args()

    # Simple walk to find all .IMA files
    pairs = []
    for root, dirs, files in os.walk(args.data_dir):
        ima_files = sorted([f for f in files if f.endswith(".IMA")])
        if len(ima_files) >= 2:
            # Assumes pairs are consecutive in sorted order
            for i in range(0, len(ima_files) - 1, 2):
                path1 = os.path.join(root, ima_files[i])
                path2 = os.path.join(root, ima_files[i + 1])
                pairs.append(f"{path1} {path2}\n")

    out_path = os.path.join(args.output, "paired_dicom.txt")
    with open(out_path, "w") as f:
        f.writelines(pairs)

    print(f"Written {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()
