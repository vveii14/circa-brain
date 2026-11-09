"""
ADNI FDG-PET dataset for image / ROI / fusion training.

- ROI source: FDG.csv + disease label.xlsx (via load_adni_fdg_pet).
- Image source: preprocessed 128^3 .npy volumes keyed by IID.
  Expected dir (set in config adni_image_dir): e.g. .../adni_fdg_pet/adni_image_128_crop_pad_rot180
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .adni_fdg_pet import load_adni_fdg_pet


_ADNI_CACHE: Dict[str, Dict[str, object]] = {}


def _resolve_image_map(image_dir: Path) -> Dict[str, str]:
    """
    Build IID -> volume path mapping.
    Prefer meta.json (if exists), fallback to <IID>.npy convention.
    """
    image_map: Dict[str, str] = {}
    meta_path = image_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for row in meta:
            iid = str(row.get("IID", ""))
            p = str(row.get("vol_filepath", ""))
            if iid and p and Path(p).exists():
                image_map[iid] = p
    if not image_map:
        for p in image_dir.glob("*.npy"):
            image_map[p.stem] = str(p)
    return image_map


def _cache_key(data_dir: Path, image_dir: Optional[Path], require_image: bool) -> str:
    return f"{data_dir.resolve()}|{str(image_dir.resolve()) if image_dir else 'None'}|{int(require_image)}"


def _load_adni_joint(
    data_dir: Path,
    image_dir: Optional[Path],
    require_image: bool,
) -> Dict[str, object]:
    """
    Load ADNI ROI and optional image paths, optionally filtering to subjects with image.
    Returns dict with keys: iids, labels, roi_vectors, roi_matrices, image_paths.
    """
    key = _cache_key(data_dir, image_dir, require_image)
    if key in _ADNI_CACHE:
        return _ADNI_CACHE[key]

    roi_vectors, labels, iids, roi_matrices = load_adni_fdg_pet(str(data_dir))
    image_map: Dict[str, str] = {}
    if image_dir is not None:
        image_map = _resolve_image_map(image_dir)

    keep_indices: List[int] = []
    image_paths: List[Optional[str]] = []
    for i, iid in enumerate(iids):
        img_p = image_map.get(str(iid))
        if require_image and img_p is None:
            continue
        keep_indices.append(i)
        image_paths.append(img_p)

    keep = np.asarray(keep_indices, dtype=np.int64)
    out: Dict[str, object] = {
        "iids": [str(iids[i]) for i in keep_indices],
        "labels": labels[keep],
        "roi_vectors": roi_vectors[keep],
        "roi_matrices": roi_matrices[keep],
        "image_paths": image_paths,
    }
    _ADNI_CACHE[key] = out
    return out


class ADNIFDGPETDataset(Dataset):
    """
    ADNI dataset that can serve:
    - image branch: preprocessed image volume (.npy, expected 128^3)
    - ROI branch: ROI vector (90,) or ROI matrix (90,90) for BrainNet
    """

    def __init__(
        self,
        data_dir: str,
        use_image: bool = True,
        use_roi: bool = True,
        n_rois: int = 90,
        indices: Optional[list] = None,
        image_dir: Optional[str] = None,
        roi_matrices: bool = True,
    ):
        data_dir_p = Path(data_dir)
        image_dir_p = Path(image_dir) if image_dir else None
        joint = _load_adni_joint(
            data_dir=data_dir_p,
            image_dir=image_dir_p,
            require_image=use_image,
        )

        self.iids: List[str] = list(joint["iids"])  # type: ignore[arg-type]
        self.labels = np.asarray(joint["labels"], dtype=np.int64)  # type: ignore[arg-type]
        self.roi_vectors = np.asarray(joint["roi_vectors"], dtype=np.float32)  # type: ignore[arg-type]
        self.roi_matrices = np.asarray(joint["roi_matrices"], dtype=np.float32)  # type: ignore[arg-type]
        self.image_paths: List[Optional[str]] = list(joint["image_paths"])  # type: ignore[arg-type]

        if self.roi_vectors.shape[1] != n_rois:
            raise ValueError(
                f"ADNI n_rois mismatch: roi_vectors.shape={self.roi_vectors.shape}, expected n_rois={n_rois}"
            )
        if self.roi_matrices.shape[1] != n_rois or self.roi_matrices.shape[2] != n_rois:
            raise ValueError(
                f"ADNI roi_matrices mismatch: shape={self.roi_matrices.shape}, expected (*,{n_rois},{n_rois})"
            )

        self.indices = indices if indices is not None else list(range(len(self.labels)))
        self.use_image = use_image
        self.use_roi = use_roi
        self.roi_as_matrix = roi_matrices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        out = {"label": int(self.labels[i])}

        if self.use_image:
            p = self.image_paths[i]
            if p is None:
                raise FileNotFoundError(f"Missing image path for IID={self.iids[i]}")
            img = np.load(p).astype(np.float32)
            # Sanitize and min-max normalize to [0, 1] per volume (ADNI .npy are raw uptake values)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 1e-6:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = img - img_min  # constant/degenerate volume -> all zeros
            if img.ndim == 3:
                img = img[np.newaxis, ...]  # (1, D, H, W)
            out["image"] = torch.from_numpy(img)

        if self.use_roi:
            if self.roi_as_matrix:
                out["roi_data"] = torch.from_numpy(self.roi_matrices[i].astype(np.float32))
            else:
                out["roi_data"] = torch.from_numpy(self.roi_vectors[i].astype(np.float32))

        return out


def get_adni_meta(
    data_dir: str,
    require_image: bool = False,
    image_dir: Optional[str] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    Return (indices, labels) for StratifiedKFold.
    If require_image=True, only keep subjects that have preprocessed image files.
    """
    data_dir_p = Path(data_dir)
    image_dir_p = Path(image_dir) if image_dir else None
    joint = _load_adni_joint(
        data_dir=data_dir_p,
        image_dir=image_dir_p,
        require_image=require_image,
    )
    labels = np.asarray(joint["labels"], dtype=np.int64)  # type: ignore[arg-type]
    return list(range(len(labels))), labels
