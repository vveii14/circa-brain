"""
Brain-structure (radiata-ai/brain-structure) dataset: 3D T1 + atlas-based ROI vector.
Expects preprocessed cache from scripts/preprocess_brain_structure.py.

Supports two modes:
- Single split: cache_dir, split="train", optional indices (into that split).
- Pooled meta: cache_dir, meta_list=[{"split","roi_idx","nii_filepath","label"}, ...], optional indices (into meta_list).
"""

import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
    import torch.nn.functional as F
except ImportError:
    nib = None
    F = None


def _load_volume_crop_resize(
    nii_path: str,
    target_shape: tuple = (96, 96, 96),
) -> np.ndarray:
    """Load 3D NIfTI, crop, normalize, resize to target_shape. Returns (1, D, H, W)."""
    if nib is None or F is None:
        raise ImportError("nibabel and torch required")
    vol = nib.load(nii_path).get_fdata()
    vol = np.asarray(vol, dtype=np.float32)
    vol = vol + np.abs(vol.min())
    if vol.max() > 0:
        vol = vol / vol.max()
    t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    d, h, w = vol.shape
    td, th, tw = target_shape
    scale = (td / d, th / h, tw / w)
    t = F.interpolate(t, scale_factor=scale, mode="trilinear", align_corners=False)
    _, _, d2, h2, w2 = t.shape
    pad_d = max(0, td - d2)
    pad_h = max(0, th - h2)
    pad_w = max(0, tw - w2)
    if pad_d or pad_h or pad_w:
        t = F.pad(
            t,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2),
        )
    t = t.squeeze(0).numpy()  # (1, D, H, W) for one channel
    return t


class BrainStructureDataset(Dataset):
    """
    Loads from preprocessed cache: ROI vectors (roi/{i}.npy) + meta (nii_filepath, label).
    Image branch: load NIfTI on the fly and resize to target_shape.

    Modes:
    - Single split: pass cache_dir, split="train", optional indices.
    - Pooled (train+val+test): pass cache_dir, meta_list=[{split, roi_idx, nii_filepath, label}, ...], optional indices.
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        use_image: bool = True,
        use_roi: bool = True,
        target_shape: tuple = (96, 96, 96),
        indices: Optional[list] = None,
        meta_list: Optional[List[dict]] = None,
    ):
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.use_image = use_image
        self.use_roi = use_roi
        self.target_shape = target_shape

        if meta_list is not None:
            # Pooled mode: meta_list is list of {"split", "roi_idx", "nii_filepath", "label"}
            full = meta_list
            if indices is not None:
                self.meta = [full[i] for i in indices]
            else:
                self.meta = full
        else:
            # Single-split mode
            meta_path = cache_dir / split / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Run scripts/preprocess_brain_structure.py first. Missing {meta_path}"
                )
            with open(meta_path) as f:
                full_meta = json.load(f)
            if indices is not None:
                self.meta = [full_meta[i] for i in indices]
                self._roi_indices = indices
            else:
                self.meta = full_meta
                self._roi_indices = list(range(len(full_meta)))
            self.roi_dir = cache_dir / split / "roi"

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        entry = self.meta[idx]
        label = entry["label"]
        out = {"label": label}
        if self.use_roi:
            if "split" in entry and "roi_idx" in entry:
                roi_path = self.cache_dir / entry["split"] / "roi" / f"{entry['roi_idx']}.npy"
            else:
                roi_idx = self._roi_indices[idx]
                roi_path = self.roi_dir / f"{roi_idx}.npy"
            roi = np.load(roi_path).astype(np.float32)
            out["roi_data"] = torch.from_numpy(roi)
        if self.use_image:
            img = _load_volume_crop_resize(entry["nii_filepath"], self.target_shape)
            out["image"] = torch.from_numpy(img)
        return out
