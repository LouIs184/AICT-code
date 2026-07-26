#!/usr/bin/env python3
"""
Generate train_img.txt / valid_img.txt for image-domain (PhaseCT) training.

Each line::
    <input_path> <label_path> <slice_index>

Usage:
    python prepare_data.py --data-dir /path/to/UIH_data --output ./
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Prepare PhaseCT data index files.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing patient subfolders.")
    parser.add_argument("--output", "-o", type=str, default="./",
                        help="Output directory for .txt files.")
    parser.add_argument("--train-patients", type=str, nargs="+", default=None,
                        help="Patient folder names for training set. Default: match original AICT.")
    parser.add_argument("--valid-patients", type=str, nargs="+", default=None,
                        help="Patient folder names for validation set. Default: match original AICT.")
    parser.add_argument("--num-slices", type=int, default=158,
                        help="Number of slices per patient (default: 158, from index 1).")
    parser.add_argument("--slice-start", type=int, default=1,
                        help="Starting slice index (default: 1, to skip boundaries).")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Default patient split from the original AICT project (this machine's actual data)
    default_train = [
        "11034_307811_960+_AXIAL_CE1_M067Y_20211215_Thick1_Incre1",
        "11849_308361_960+_AXIAL_CE1_M044Y_20211216_Thick1_Incre1",
        "15962_15633_960+_AXIAL_CE1_M034Y_20200601_Thick1_Incre1",
        "15962_35868_960+_AXIAL_CE1_M034Y_20200601_Thick1_Incre1",
        "18631_303226_960+_AXIAL_CE1_F058Y_20211216_Thick1_Incre1",
        "38423_35948_960+_AXIAL_CE1_M049Y_20200601_Thick1_Incre1",
        "51741_308269_960+_AXIAL_CE1_M048Y_20211215_Thick1_Incre1",
        "51789_393117_960+_AXIAL_CE1_M074Y_20211215_Thick1_Incre1",
        "72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1",
        "91963_301643_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1",
        "91963_53624_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1",
    ]
    default_valid = [
        "17238_33314_960+_AXIAL_CE1_F063Y_20220106_Thick1_Incre1",
        "18537_93073_960+_AXIAL_CE0_M067Y_20210430_Thick1_Incre1",
    ]

    train_patients = args.train_patients or default_train
    valid_patients = args.valid_patients or default_valid

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
