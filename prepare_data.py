#!/usr/bin/env python3
"""
Generate train_img.txt / valid_img.txt for image-domain (PhaseCT) training.

Each line::
    <input_path> <label_path> <slice_index>

The train/valid patient split is read from patients.yaml (--patients-config).
CLI --train-patients / --valid-patients override it when given.

Usage:
    python prepare_data.py --data-dir /path/to/UIH_data --output ./
    python prepare_data.py --data-dir /path/to/UIH_data --output ./ \
        --patients-config my_split.yaml
"""

import argparse
import os
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.io_utils import load_patients


def main():
    parser = argparse.ArgumentParser(description="Prepare PhaseCT data index files.")
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
    parser.add_argument("--num-slices", type=int, default=158,
                        help="Number of slices per patient (default: 158, from index 1).")
    parser.add_argument("--slice-start", type=int, default=1,
                        help="Starting slice index (default: 1, to skip boundaries).")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Train/valid split comes from patients.yaml (single source of truth).
    train_patients, valid_patients = load_patients(args.patients_config)
    if args.train_patients is not None:
        train_patients = args.train_patients
    if args.valid_patients is not None:
        valid_patients = args.valid_patients

    def write_list(filename, patients, data_dir):
        with open(os.path.join(args.output, filename), "w", encoding="utf-8") as f:
            for patient in patients:
                input_path = os.path.join(
                    data_dir, patient, "rec_pbi_fs_dec_blur_phase_100000.raw"
                )
                label_path = os.path.join(data_dir, patient, "rec_no_pbi.raw")
                for idx in range(args.slice_start, args.slice_start + args.num_slices):
                    f.write(f"{input_path} {label_path} {idx}\n")
        print(f"Written {filename} with {len(patients)} patients x {args.num_slices} slices.")

    write_list("train_img.txt", train_patients, args.data_dir)
    write_list("valid_img.txt", valid_patients, args.data_dir)

    print("Done.")


if __name__ == "__main__":
    main()
