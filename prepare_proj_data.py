#!/usr/bin/env python3
"""
Generate train_proj_img.txt / valid_proj_img.txt for projection-domain (ProjCT) training.

Each line::
    <input_path> <label_path> <view_index>

Views range from 0 to 719 (720 views total).

The train/valid patient split is read from patients.yaml (--patients-config).
CLI --train-patients / --valid-patients override it when given.

Usage:
    python prepare_proj_data.py --data-dir /path/to/UIH_data --output ./
    python prepare_proj_data.py --data-dir /path/to/UIH_data --output ./ \
        --patients-config my_split.yaml
"""

import argparse
import os
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.io_utils import load_patients


def main():
    parser = argparse.ArgumentParser(description="Prepare ProjCT data index files.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing patient subfolders.")
    parser.add_argument("--output", "-o", type=str, default="./",
                        help="Output directory for .txt files.")
    parser.add_argument("--patients-config", type=str, default="patients.yaml",
                        help="YAML file with 'train'/'valid' patient lists. "
                             "Default: ./patients.yaml")
    parser.add_argument("--train-patients", type=str, nargs="+", default=None,
                        help="Override the train patient list (from patients.yaml).")
    parser.add_argument("--valid-patients", type=str, nargs="+", default=None,
                        help="Override the valid patient list (from patients.yaml).")
    parser.add_argument("--num-views", type=int, default=720)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Train/valid split comes from patients.yaml (single source of truth).
    # All domains (PhaseCT, ProjCT, etc.) use the same split.
    train_patients, valid_patients = load_patients(args.patients_config)
    if args.train_patients is not None:
        train_patients = args.train_patients
    if args.valid_patients is not None:
        valid_patients = args.valid_patients

    def write_list(filename, patients):
        with open(os.path.join(args.output, filename), "w") as f:
            for patient in patients:
                input_path = os.path.join(
                    args.data_dir, patient, "proj_pbi_fs_dec_blur_phase_100000.raw"
                )
                label_path = os.path.join(args.data_dir, patient, "proj_no_pbi.raw")
                for view in range(args.num_views):
                    f.write(f"{input_path} {label_path} {view}\n")
        print(f"Written {filename} with {len(patients)} patients x {args.num_views} views.")

    write_list("train_proj_img.txt", train_patients)
    write_list("valid_proj_img.txt", valid_patients)

    print("Done.")


if __name__ == "__main__":
    main()
