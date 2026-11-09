"""
ADHD dataset: 5 npy folds with raw_img_np (96,96,96), raw_img_aal_mask_np, label.
See /mnt/wel425/project/miccai26/data/adhd/docs.md.
ROI = per-AAL-region mean of raw_img -> (n_rois,) vector.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Cache full concatenated data by data_dir to avoid reloading for train/val
_ADHD_CACHE: dict = {}


def _load_adhd_all(
    data_dir: Path,
    n_rois: Optional[int] = None,
    precompute_roi: bool = False,
    fold_prefix: str = "adhd",
    fold_suffix: str = "775",
):
    """Load 5 folds (cached once); if precompute_roi, compute and return (data, roi_vectors).
    fold_prefix/fold_suffix: e.g. ('adhd','775') -> adhd_fold1_775.npy; ('abide','1052') -> abide_fold1_1052.npy.
    """
    key_data = f"{data_dir}|{fold_prefix}|{fold_suffix}"
    if key_data not in _ADHD_CACHE:
        arrs = []
        for k in range(1, 6):
            p = data_dir / f"{fold_prefix}_fold{k}_{fold_suffix}.npy"
            if not p.exists():
                raise FileNotFoundError(f"Fold not found: {p}")
            arrs.append(np.load(p, allow_pickle=True))
        _ADHD_CACHE[key_data] = np.concatenate(arrs, axis=0)
    data = _ADHD_CACHE[key_data]
    if precompute_roi and n_rois is not None:
        key_roi = f"{key_data}_roi{n_rois}"
        if key_roi not in _ADHD_CACHE:
            roi_arr = np.zeros((len(data), n_rois), dtype=np.float32)
            for i in range(len(data)):
                raw = np.asarray(data[i]["raw_img_np"], dtype=np.float32)
                aal = np.asarray(data[i]["raw_img_aal_mask_np"], dtype=np.float32)
                roi_arr[i] = _aal_mask_to_roi_vector(raw, aal, n_rois)
            _ADHD_CACHE[key_roi] = roi_arr
        return (data, _ADHD_CACHE[key_roi])
    return data


def _aal_mask_to_roi_vector(raw_img: np.ndarray, aal_mask: np.ndarray, n_rois: int) -> np.ndarray:
    """Compute per-ROI mean intensity (vectorized with bincount).
    AAL mask may use 1..n_rois or non-consecutive codes (e.g. 2001, 2101, ..., 9170); we remap to 1..n_rois.
    """
    raw_img = np.asarray(raw_img, dtype=np.float32).ravel()
    aal_mask = np.asarray(aal_mask, dtype=np.float64).ravel()
    unique = np.unique(aal_mask)
    unique_nonzero = unique[unique > 0]
    if len(unique_nonzero) == 0:
        return np.zeros(n_rois, dtype=np.float32)
    # If mask already uses 1..n_rois (and only those), use bincount directly
    if len(unique_nonzero) <= n_rois and unique_nonzero.max() <= n_rois and unique_nonzero.min() >= 1:
        aal_int = aal_mask.astype(np.int32)
        max_lab = min(int(aal_int.max()) + 1, n_rois + 1)
        sum_by = np.bincount(aal_int, weights=raw_img, minlength=max_lab + 1)
        cnt_by = np.bincount(aal_int, minlength=max_lab + 1)
        roi_vec = np.zeros(n_rois, dtype=np.float32)
        for i in range(1, min(n_rois + 1, len(cnt_by))):
            if cnt_by[i] > 0:
                roi_vec[i - 1] = sum_by[i] / cnt_by[i]
        return roi_vec
    # Remap non-consecutive AAL codes (e.g. 2001..9170) to 1..n_rois
    n_actual = min(len(unique_nonzero), n_rois)
    old_to_new = {int(old): (new + 1) for new, old in enumerate(sorted(unique_nonzero)[:n_rois])}
    new_mask = np.zeros_like(aal_mask, dtype=np.int32)
    for old_lab, new_lab in old_to_new.items():
        new_mask[aal_mask == old_lab] = new_lab
    max_lab = n_actual + 1
    sum_by = np.bincount(new_mask, weights=raw_img, minlength=max_lab + 1)
    cnt_by = np.bincount(new_mask, minlength=max_lab + 1)
    roi_vec = np.zeros(n_rois, dtype=np.float32)
    for i in range(1, n_actual + 1):
        if cnt_by[i] > 0:
            roi_vec[i - 1] = sum_by[i] / cnt_by[i]
    return roi_vec


# Default filename for precomputed ROI vectors (from scripts/precompute_adhd_roi.py)
ADHD_ROI_VECTORS_FILE = "roi_vectors_775.npy"
# Precomputed Pearson 116×116 matrices (from scripts/precompute_adhd_roi_matrices.py)
ADHD_ROI_MATRICES_FILE = "roi_matrices_775.npy"


class ADHDDataset(Dataset):
    """
    ADHD: raw_img_np (96,96,96), label; ROI from roi_vectors_775.npy, roi_matrices_775.npy,
    or computed on the fly. For ROI-based (NeuroGraph) with 116×116 Pearson matrices, set roi_matrices_file.
    """

    def __init__(
        self,
        data_dir: str,
        use_image: bool = True,
        use_roi: bool = True,
        n_rois: int = 90,
        indices: Optional[list] = None,
        roi_vectors_file: Optional[str] = None,
        roi_matrices_file: Optional[str] = None,
        fold_prefix: str = "adhd",
        fold_suffix: str = "775",
    ):
        data_dir = Path(data_dir)
        self._data = _load_adhd_all(data_dir, fold_prefix=fold_prefix, fold_suffix=fold_suffix)
        self._roi_vec = None
        self._roi_mat = None  # (N, n_rois, n_rois) when loading precomputed matrices
        if use_roi:
            if roi_matrices_file:
                mat_path = data_dir / roi_matrices_file
                if mat_path.exists():
                    self._roi_mat = np.load(mat_path).astype(np.float32)
                    assert self._roi_mat.ndim == 3 and self._roi_mat.shape[0] == len(self._data), (
                        f"roi_matrices shape {self._roi_mat.shape} vs data len {len(self._data)}"
                    )
                    assert self._roi_mat.shape[1] == self._roi_mat.shape[2] == n_rois, (
                        f"roi_matrices {self._roi_mat.shape[1]}x{self._roi_mat.shape[2]} vs n_rois {n_rois}"
                    )
            if self._roi_mat is None:
                roi_path = data_dir / (roi_vectors_file or ADHD_ROI_VECTORS_FILE)
                if roi_path.exists():
                    self._roi_vec = np.load(roi_path).astype(np.float32)
                    assert self._roi_vec.shape[0] == len(self._data) and self._roi_vec.shape[1] == n_rois, (
                        f"roi_vectors shape {self._roi_vec.shape} vs data {len(self._data)} n_rois {n_rois}"
                    )
                else:
                    _, self._roi_vec = _load_adhd_all(
                        data_dir, n_rois=n_rois, precompute_roi=True,
                        fold_prefix=fold_prefix, fold_suffix=fold_suffix,
                    )
        self.indices = indices if indices is not None else list(range(len(self._data)))
        self.use_image = use_image
        self.use_roi = use_roi
        self.n_rois = n_rois

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        row = self._data[i]
        label = int(row["label"])
        out = {"label": label}
        if self.use_image:
            img = np.asarray(row["raw_img_np"], dtype=np.float32)
            if img.ndim == 3:
                img = img[np.newaxis, ...]
            out["image"] = torch.from_numpy(img)
        if self.use_roi:
            if self._roi_mat is not None:
                out["roi_data"] = torch.from_numpy(self._roi_mat[i].astype(np.float32))  # (n_rois, n_rois)
            elif self._roi_vec is not None:
                out["roi_data"] = torch.from_numpy(self._roi_vec[i].astype(np.float32))
            else:
                raw = np.asarray(row["raw_img_np"], dtype=np.float32)
                aal = np.asarray(row["raw_img_aal_mask_np"], dtype=np.float32)
                out["roi_data"] = torch.from_numpy(_aal_mask_to_roi_vector(raw, aal, self.n_rois))
        return out


def get_adhd_meta(
    data_dir: str,
    fold_prefix: str = "adhd",
    fold_suffix: str = "775",
) -> Tuple[List[int], np.ndarray]:
    """Return (list of indices 0..N-1, labels array) for StratifiedKFold."""
    data_dir = Path(data_dir)
    data = _load_adhd_all(data_dir, fold_prefix=fold_prefix, fold_suffix=fold_suffix)
    n = len(data)
    labels = np.array([int(data[i]["label"]) for i in range(n)])
    return list(range(n)), labels
