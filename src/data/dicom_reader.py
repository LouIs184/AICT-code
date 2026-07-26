"""
DICOM file reader for pairs of .IMA slices.

The index file contains paired DICOM file paths:
    ldct_path rdct_path

where each line contains the full path to the low-dose and regular-dose DICOM files.
"""

import pydicom
import locale
import torch
import numpy as np
from torch.utils.data import Dataset


class dicom_reader(Dataset):
    """Dataset for paired DICOM CT slices."""

    def __init__(self, paired_img_txt):
        super().__init__()
        try:
            with open(paired_img_txt, encoding="utf-8") as f:
                self.paired_files = f.readlines()
        except UnicodeDecodeError:
            with open(paired_img_txt, encoding=locale.getpreferredencoding()) as f:
                self.paired_files = f.readlines()

    def __len__(self):
        return len(self.paired_files)

    @staticmethod
    def to_tensor(data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            data = data.transpose(2, 0, 1)
        return torch.FloatTensor(data)

    @staticmethod
    def to_numpy(data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)
        return data

    def _get(self, paired_str):
        # Parse line: "ldct.IMA rdct.IMA"
        parts = paired_str.strip().split()
        ldct_path, rdct_path = parts[0], parts[-1]

        ldct_img = pydicom.read_file(ldct_path).pixel_array
        ldct_img = np.maximum(ldct_img, 0)

        rdct_img = pydicom.read_file(rdct_path).pixel_array
        rdct_img = np.maximum(rdct_img, 0)

        ldct_img = ldct_img.astype(np.float32)
        rdct_img = rdct_img.astype(np.float32)

        return {
            "ldct": self.to_tensor(ldct_img),
            "rdct": self.to_tensor(rdct_img),
        }

    def __getitem__(self, index):
        return self._get(self.paired_files[index])
