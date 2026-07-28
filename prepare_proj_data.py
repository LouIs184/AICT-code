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

    # Default split — MUST match prepare_data.py (image domain) exactly.
    # All domains (PhaseCT, ProjCT, etc.) must use the same patient split.
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
