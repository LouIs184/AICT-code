#!/usr/bin/env python3
"""
Generate train_proj_img.txt / valid_proj_img.txt for projection-domain (ProjCT) training.

Each line::
    <input_path> <label_path> <view_index>

Views range from 0 to 719 (720 views total).

Usage:
    python prepare_proj_data.py --data-dir /path/to/UIH_data --output ./
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Prepare ProjCT data index files.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing patient subfolders.")
    parser.add_argument("--output", "-o", type=str, default="./",
                        help="Output directory for .txt files.")
    parser.add_argument("--train-patients", type=str, nargs="+", default=None)
    parser.add_argument("--valid-patients", type=str, nargs="+", default=None)
    parser.add_argument("--num-views", type=int, default=720)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Default split (same patients as PhaseCT)
    default_train = [
        "17424_960+_AXIAL_CE1_M058Y_20221101_Thick1_Incre1",
        "18437_1005_960+_AXIAL_CE1_F059Y_20220406_Thick1_Incre1",
        "18847_7779_960+_AXIAL_CE1_F060Y_20220111_Thick1_Incre1",
        "19824_30332_960+_AXIAL_CE1_M059Y_20220915_Thick1_Incre1",
        "17238_33314_960+_AXIAL_CE1_F063Y_20220106_Thick1_Incre1",
        "38423_35948_960+_AXIAL_CE1_M049Y_20200601_Thick1_Incre1",
        "28637_1359_960+_AXIAL_CE1_F062Y_20200413_Thick1_Incre1",
        "2116_1770_960+_AXIAL_CE1_F066Y_20200527_Thick1_Incre1",
        "2_20743_960+_AXIAL_CE1_F040Y_20200508_Thick1_Incre1",
        "26076_2484_960+_AXIAL_CE1_F064Y_20200331_Thick1_Incre1",
        "18537_23851_960+_AXIAL_CE1_F062Y_20200402_Thick1_Incre1",
    ]
    default_valid = [
        "17238_33314_960+_AXIAL_CE1_F063Y_20220106_Thick1_Incre1",
        "38423_35948_960+_AXIAL_CE1_M049Y_20200601_Thick1_Incre1",
    ]

    train_patients = args.train_patients or default_train
    valid_patients = args.valid_patients or default_valid

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
