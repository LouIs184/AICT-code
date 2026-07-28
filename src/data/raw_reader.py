"""
Raw float32 volume data reader (PhaseCT / image domain).

Reads CT slice pairs from raw float32 volume files.
Data shape per volume: (160, 512, 512).

Normalization: HU range [-1000, +1000] → [0, 1].
"""

import locale
import torch
import numpy as np
from torch.utils.data import Dataset


class raw_reader(Dataset):
    """
    Dataset for raw CT volume slices (image domain).

    Each line in the index file has format::
        input_path label_path slice_index

    where input_path / label_path point to .raw files of shape (160, 512, 512).
    """

    def __init__(self, paired_img_txt, hu_min=-1000.0, hu_max=1000.0):
        super().__init__()
        try:
            with open(paired_img_txt, encoding="utf-8") as f:
                self.paired_files = f.readlines()
        except UnicodeDecodeError:
            with open(paired_img_txt, encoding=locale.getpreferredencoding()) as f:
                self.paired_files = f.readlines()
        self.hu_min = hu_min
        self.hu_range = hu_max - hu_min

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, index):
        line = self.paired_files[index].strip()
        input_path, label_path, slice_idx = line.split()
        slice_idx = int(slice_idx)

        # Read only the target slice from each volume
        input_data = np.fromfile(input_path, dtype=np.float32).reshape(160, 512, 512)
        label_data = np.fromfile(label_path, dtype=np.float32).reshape(160, 512, 512)

        input_slice = input_data[slice_idx].astype(np.float32)
        label_slice = label_data[slice_idx].astype(np.float32)

        # HU normalization: [hu_min, hu_max] → [0, 1]
        input_slice = (input_slice - self.hu_min) / self.hu_range
        label_slice = (label_slice - self.hu_min) / self.hu_range

        # Clip outliers
        input_slice = np.clip(input_slice, 0, 1)
        label_slice = np.clip(label_slice, 0, 1)

        return {
            "ldct": torch.FloatTensor(input_slice[np.newaxis, ...]),
            "rdct": torch.FloatTensor(label_slice[np.newaxis, ...]),
        }
