"""
Subject-level dataset for the imaging + ROI-connectivity model (ADHD-200 / ABIDE).

Expected files in `data_dir` (`<prefix>` is e.g. "adhd" or "abide", `<N>` the subject count):

    <prefix>_fold1_<N>.npy ... <prefix>_fold5_<N>.npy   subject records (see README)
    roi_matrices_<N>.npy                                 (N, n_rois, n_rois) connectivity matrices

The five fold files are concatenated in order 1..5; row i of roi_matrices_<N>.npy must belong to
subject i of that concatenation. `<N>` is read from the fold-file names.

Demo mode reads the bundled synthetic samples instead (random data, for smoke tests only):

    <prefix>_demo10.npy, roi_matrices_demo10.npy
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_CACHE: dict = {}


def resolve_files(data_dir, prefix: str = "adhd", demo: bool = False) -> Tuple[List[Path], Path]:
    """Return (subject files in concatenation order, default ROI-matrix file)."""
    data_dir = Path(data_dir)
    if demo:
        return [data_dir / f"{prefix}_demo10.npy"], data_dir / "roi_matrices_demo10.npy"
    first = sorted(data_dir.glob(f"{prefix}_fold1_*.npy"))
    if len(first) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{prefix}_fold1_<N>.npy' in {data_dir}, found {[p.name for p in first]}. "
            "Use --demo to run on the bundled 10-subject subset."
        )
    suffix = first[0].stem[len(f"{prefix}_fold1_"):]
    files = [data_dir / f"{prefix}_fold{k}_{suffix}.npy" for k in range(1, 6)]
    return files, data_dir / f"roi_matrices_{suffix}.npy"


def _load_subjects(files: List[Path]) -> np.ndarray:
    key = tuple(str(p) for p in files)
    if key not in _CACHE:
        for p in files:
            if not p.exists():
                raise FileNotFoundError(f"Missing data file: {p}")
        _CACHE[key] = np.concatenate([np.load(p, allow_pickle=True) for p in files], axis=0)
    return _CACHE[key]


def _load_matrices(path: Path, n_subjects: int, n_rois: int) -> np.ndarray:
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            raise FileNotFoundError(f"Missing ROI connectivity file: {path}")
        _CACHE[key] = np.load(path).astype(np.float32)
    mats = _CACHE[key]
    if mats.shape != (n_subjects, n_rois, n_rois):
        raise ValueError(f"{path.name} has shape {mats.shape}, expected {(n_subjects, n_rois, n_rois)}")
    return mats


class ADHDDataset(Dataset):
    """Returns {"label", "image": (1, D, H, W), "roi_data": (n_rois, n_rois)} per subject."""

    def __init__(
        self,
        data_dir: str,
        indices: Optional[List[int]] = None,
        use_image: bool = True,
        use_roi: bool = True,
        n_rois: int = 116,
        prefix: str = "adhd",
        demo: bool = False,
        roi_matrices_file: Optional[str] = None,
    ):
        files, default_roi = resolve_files(data_dir, prefix, demo)
        self._data = _load_subjects(files)
        self._roi_mat = None
        if use_roi:
            roi_path = Path(data_dir) / roi_matrices_file if roi_matrices_file else default_roi
            self._roi_mat = _load_matrices(roi_path, len(self._data), n_rois)
        self.indices = list(indices) if indices is not None else list(range(len(self._data)))
        self.use_image = use_image
        self.use_roi = use_roi

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        row = self._data[i]
        out = {"label": int(row["label"])}
        if self.use_image:
            img = np.asarray(row["raw_img_np"], dtype=np.float32)
            out["image"] = torch.from_numpy(img[np.newaxis] if img.ndim == 3 else img)
        if self.use_roi:
            out["roi_data"] = torch.from_numpy(self._roi_mat[i])
        return out


def get_adhd_meta(data_dir: str, prefix: str = "adhd", demo: bool = False) -> Tuple[List[int], np.ndarray]:
    """Return (subject indices 0..N-1, labels) of the concatenated subject list."""
    files, _ = resolve_files(data_dir, prefix, demo)
    data = _load_subjects(files)
    labels = np.array([int(data[i]["label"]) for i in range(len(data))])
    return list(range(len(data))), labels


def collate(batch: List[dict]) -> dict:
    out = {"label": torch.tensor([b["label"] for b in batch], dtype=torch.long)}
    for key in ("image", "roi_data"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    return out
