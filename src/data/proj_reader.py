"""
Projection domain data reader (ProjCT).

Reads individual views from projection-domain .raw files.
The per-view shape is configurable via ``view_shape``; the default is the
legacy (128, 512), while the NF0128 data uses (500, 500).

Normalization: clip to [0, norm_clip_max] → scale to [0, 1].
Default norm_clip_max=0.28 based on empirical data range (~ -0.066 to 0.276).
"""

import json
import locale
import os
import re

import torch
import numpy as np
from torch.utils.data import Dataset

_LAM = 1.24 / 50 * 1e-6      # mm, 50 keV
_SID = 1e6                   # mm


def _nf_from_az(a, z):
    """Fresnel number from aperture a (mm) and propagation distance z (mm)."""
    m = 1.0 + float(z) / _SID
    a_eff = float(a) / m
    z_eff = float(z) / m
    return a_eff ** 2 / (_LAM * z_eff)


def read_raw_volume(path, view_shape=(128, 512), dtype=np.float32):
    """Read a ``(views, *view_shape)`` raw volume, inferring the view count.

    Useful for inference scripts that must not hardcode the number of views
    (e.g. NF0128 has 100 views, the old test set has 720).
    """
    view_shape = tuple(int(v) for v in view_shape)
    view_size = int(np.prod(view_shape))
    item_size = view_size * np.dtype(dtype).itemsize
    size = os.path.getsize(path)
    if size % item_size != 0:
        raise ValueError(
            f"{path}: file size {size} is not a multiple of {item_size} "
            f"(view_shape={view_shape}, dtype={np.dtype(dtype)})"
        )
    n_views = size // item_size
    return np.fromfile(path, dtype=dtype).reshape((n_views,) + view_shape)


def crop_projection(a, crop_rows=0, crop_cols=0):
    """Crop rows/columns of a 2-D view or a 3-D ``(views, H, W)`` volume."""
    if crop_rows > 0:
        a = a[..., crop_rows:-crop_rows, :]
    if crop_cols > 0:
        a = a[..., :, crop_cols:-crop_cols]
    return a


class proj_reader(Dataset):
    """
    Dataset for projection-domain CT data.

    Each line in the index file has format::
        input_path label_path view_index

    Each line indexes a single view from a projection volume whose per-view
    shape is ``view_shape`` (e.g. (128, 512) for the legacy data, (500, 500)
    for NF0128).
    """

    def __init__(self, paired_img_txt, norm_clip_max=0.28, crop_rows=0,
                 crop_cols=0, view_shape=(128, 512), phys_file="",
                 heads_json="", clip_meas=False):
        super().__init__()
        try:
            with open(paired_img_txt, encoding="utf-8") as f:
                self.paired_files = f.readlines()
        except UnicodeDecodeError:
            with open(paired_img_txt, encoding=locale.getpreferredencoding()) as f:
                self.paired_files = f.readlines()
        self.norm_clip_max = norm_clip_max
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.view_shape = tuple(int(v) for v in view_shape)
        self.view_size = int(np.prod(self.view_shape))
        self.phys_file = phys_file or ""
        self.clip_meas = bool(clip_meas)

        # 头号 -> NF；用于 varNF 的 per-sample 菲涅尔数
        self.head_nf = {}
        if heads_json:
            with open(heads_json, encoding="utf-8") as f:
                table = json.load(f)
            for head, az in table.items():
                self.head_nf[str(head).zfill(2)] = _nf_from_az(az[0], az[1])

    def __len__(self):
        return len(self.paired_files)

    def _read_view(self, path, view_idx):
        byte_offset = view_idx * self.view_size * 4  # float32 = 4 bytes
        flat = np.fromfile(path, dtype=np.float32,
                           count=self.view_size, offset=byte_offset)
        if flat.size != self.view_size:
            raise ValueError(
                f"Failed to read view {view_idx} with view_shape {self.view_shape}: "
                f"got {flat.size} values instead of {self.view_size} from {path}. "
                f"Check view_shape and the raw file size."
            )
        return flat.reshape(self.view_shape)

    def _norm(self, a):
        return np.clip(a, 0, self.norm_clip_max) / self.norm_clip_max

    def _norm_meas(self, a):
        """物理目标图：默认只除以 clip_max，保留负值和 >clip_max 的过冲。

        传播后的 post-log 在暗条纹可达负值、亮条纹可超过训练 clip 上界；
        如果再 clip，目标会被截断（头 18 上截断改变量的 RMS 约为传播信号
        RMS 的 79%），物理项就在拟合一个错误的目标。只有显式设置
        ``clip_meas=True`` 时才复现旧 Restormer 的截断行为。
        """
        if self.clip_meas:
            return np.clip(a, 0, self.norm_clip_max) / self.norm_clip_max
        return a / self.norm_clip_max

    def __getitem__(self, index):
        line = self.paired_files[index].strip()
        input_path, label_path, view_idx = line.split()
        view_idx = int(view_idx)

        input_view = self._read_view(input_path, view_idx)
        label_view = self._read_view(label_path, view_idx)

        # Crop black border rows (top & bottom) and columns (left & right)
        input_view = crop_projection(input_view, self.crop_rows, self.crop_cols)
        label_view = crop_projection(label_view, self.crop_rows, self.crop_cols)

        input_view = self._norm(input_view)
        label_view = self._norm(label_view)

        out = {
            "ldct": torch.FloatTensor(input_view[np.newaxis, ...]),
            "rdct": torch.FloatTensor(label_view[np.newaxis, ...]),
        }

        # 物理一致性损失的传播目标（可选，与输入同目录）
        if self.phys_file:
            phys_path = (self.phys_file if os.path.isabs(self.phys_file)
                         else os.path.join(os.path.dirname(input_path),
                                           self.phys_file))
            meas = self._read_view(phys_path, view_idx)
            meas = crop_projection(meas, self.crop_rows, self.crop_cols)
            out["meas"] = torch.FloatTensor(
                self._norm_meas(meas)[np.newaxis, ...])

        # per-head NF（可选）
        if self.head_nf:
            m = re.search(r"/(\d{2})/dose_", input_path.replace("\\", "/"))
            if m is None:
                raise ValueError(
                    f"cannot parse head id from path: {input_path}")
            head = m.group(1)
            if head not in self.head_nf:
                raise KeyError(f"head {head} not in heads_json")
            out["nf"] = torch.tensor(self.head_nf[head], dtype=torch.float32)

        return out
