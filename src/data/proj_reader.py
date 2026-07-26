"""
Projection domain data reader (ProjCT).

Reads individual views from projection-domain .raw files.
Data shape: (720, 128, 512) = (views, detector_rows, detector_channels).

Normalization: clip to [0, norm_clip_max] → scale to [0, 1].
Default norm_clip_max=0.28 based on empirical data range (~ -0.066 to 0.276).
"""

import locale
import torch
import numpy as np
from torch.utils.data import Dataset


class proj_reader(Dataset):
    """
    Dataset for projection-domain CT data.

    Each line in the index file has format::
        input_path label_path view_index

    Each line indexes a single view from a (720, 128, 512) projection volume.
    """

    def __init__(self, paired_img_txt, norm_clip_max=0.28):
        super().__init__()
        try:
            with open(paired_img_txt, encoding="utf-8") as f:
                self.paired_files = f.readlines()
        except UnicodeDecodeError:
            with open(paired_img_txt, encoding=locale.getpreferredencoding()) as f:
                self.paired_files = f.readlines()
        self.norm_clip_max = norm_clip_max

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, index):
        line = self.paired_files[index].strip()
        input_path, label_path, view_idx = line.split()
        view_idx = int(view_idx)

        # Read only the target view (128×512) via file seek
        view_size = 128 * 512
        byte_offset = view_idx * view_size * 4  # float32 = 4 bytes

        input_view = np.fromfile(
            input_path, dtype=np.float32,
            count=view_size, offset=byte_offset
        ).reshape(128, 512)

        label_view = np.fromfile(
            label_path, dtype=np.float32,
            count=view_size, offset=byte_offset
        ).reshape(128, 512)

        # Normalize to [0, 1]
        input_view = np.clip(input_view, 0, self.norm_clip_max) / self.norm_clip_max
        label_view = np.clip(label_view, 0, self.norm_clip_max) / self.norm_clip_max

        return {
            "ldct": torch.FloatTensor(input_view[np.newaxis, ...]),
            "rdct": torch.FloatTensor(label_view[np.newaxis, ...]),
        }
