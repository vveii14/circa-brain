"""
fMRI dataset: 4D volume + ROI timeseries/connectivity.
Supports NIfTI + atlas for extraction, or pre-extracted ROI data.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
    from nilearn import datasets, maskers
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False


def extract_roi_timeseries(
    fmri_img,
    atlas_name: str = "schaefer_100",
    n_rois: int = 100,
    atlas_path: Optional[str] = None,
) -> np.ndarray:
    """Extract ROI mean timeseries from 4D fMRI using atlas. Shape (T, n_rois)."""
    if not HAS_NILEARN:
        raise ImportError("nilearn required for ROI extraction. pip install nilearn")
    if atlas_path and os.path.isfile(atlas_path):
        masker = maskers.NiftiLabelsMasker(
            labels_img=atlas_path,
            standardize=True,
            memory="nilearn_cache",
            verbose=0,
        )
    else:
        if atlas_name == "schaefer_100":
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
            labels_img = atlas.maps
        else:
            raise ValueError(f"Unknown atlas: {atlas_name}. Provide atlas_path.")
        masker = maskers.NiftiLabelsMasker(
            labels_img=labels_img,
            standardize=True,
            memory="nilearn_cache",
            verbose=0,
        )
    return masker.fit_transform(fmri_img)  # (T, n_rois)


def compute_connectivity(timeseries: np.ndarray) -> np.ndarray:
    """Pearson correlation -> (n_rois, n_rois). NaN/Inf set to 0."""
    corr = np.corrcoef(timeseries.T)
    if not np.isfinite(corr).all():
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32)


class fMRIImageROIDataset(Dataset):
    """
    Returns:
        image: (C, D, H, W) or (C, T, D, H, W) — 3D volume or 4D for temporal handling
        roi_data: (n_rois, n_rois) connectivity or (T, n_rois) timeseries
        label: int
    """

    def __init__(
        self,
        samples: list,
        roi_format: str = "connectivity",  # "connectivity" | "timeseries"
        use_image: bool = True,
        use_roi: bool = True,
        atlas_name: str = "schaefer_100",
        n_rois: int = 100,
        atlas_path: Optional[str] = None,
        temporal_pool: str = "mean",  # "mean" | "last" | "none"
        device: Optional[torch.device] = None,
    ):
        """
        samples: list of dicts with keys:
            - "fmri": path to 4D NIfTI, or "roi_timeseries": (T, n_rois) array, or "roi_connectivity": (n_rois, n_rois)
            - "label": int
        """
        self.samples = samples
        self.roi_format = roi_format
        self.use_image = use_image
        self.use_roi = use_roi
        self.atlas_name = atlas_name
        self.n_rois = n_rois
        self.atlas_path = atlas_path
        self.temporal_pool = temporal_pool
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    def _load_volume(self, path: str) -> np.ndarray:
        if path.endswith(".npy"):
            data = np.load(path)
        else:
            img = nib.load(path)
            data = np.asarray(img.dataobj)
        if data.ndim == 4:
            if self.temporal_pool == "mean":
                data = np.mean(data, axis=0)
            elif self.temporal_pool == "last":
                data = data[-1]
            # else keep 4D
        return data.astype(np.float32)

    def _get_roi_from_fmri(self, fmri_path: str) -> np.ndarray:
        if fmri_path.endswith(".npy"):
            data = np.load(fmri_path)
            if data.ndim == 4:
                ts = np.mean(data, axis=(1, 2, 3))  # dummy: no atlas, fake (T,) -> broadcast to (T, n_rois)
                ts = np.broadcast_to(ts[:, None], (ts.shape[0], self.n_rois))
            else:
                ts = data
            return compute_connectivity(ts) if self.roi_format == "connectivity" else ts.astype(np.float32)
        fmri_img = nib.load(fmri_path)
        ts = extract_roi_timeseries(
            fmri_img,
            atlas_name=self.atlas_name,
            n_rois=self.n_rois,
            atlas_path=self.atlas_path,
        )
        if self.roi_format == "connectivity":
            return compute_connectivity(ts)
        return ts

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        label = int(s["label"])
        roi_data = None
        image = None

        if self.use_roi:
            if "roi_connectivity" in s:
                roi_data = np.asarray(s["roi_connectivity"], dtype=np.float32)
            elif "roi_timeseries" in s:
                ts = np.asarray(s["roi_timeseries"], dtype=np.float32)
                roi_data = compute_connectivity(ts) if self.roi_format == "connectivity" else ts
            elif "fmri" in s:
                roi_data = self._get_roi_from_fmri(s["fmri"])

        if self.use_image and "fmri" in s:
            image = self._load_volume(s["fmri"])
            if image.ndim == 3:
                image = image[np.newaxis, ...]  # (1, D, H, W)

        out = {"label": label}
        if image is not None:
            out["image"] = torch.from_numpy(image)
        if roi_data is not None:
            out["roi_data"] = torch.from_numpy(roi_data)
        return out
