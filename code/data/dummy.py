"""
Generate dummy 4D fMRI and ROI data for testing the pipeline without real NIfTI.
"""

import numpy as np
from pathlib import Path


def get_dummy_fmri_data(
    data_dir: str = "./data/dummy",
    n_subjects: int = 40,
    n_classes: int = 2,
    shape_4d: tuple = (50, 40, 40, 40),
    n_rois: int = 100,
):
    """
    Create dummy dataset: random 4D volumes + random ROI timeseries.
    Saves as .npy for volume and ROI for speed; no NIfTI written by default.
    Returns list of sample dicts for fMRIImageROIDataset.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    samples = []
    T, D, H, W = shape_4d
    for i in range(n_subjects):
        label = i % n_classes
        # Dummy 4D fMRI
        vol = np.random.randn(*shape_4d).astype(np.float32) * 0.5
        roi_ts = np.random.randn(T, n_rois).astype(np.float32) * 0.5
        conn = np.corrcoef(roi_ts.T)
        np.save(Path(data_dir) / f"sub_{i:03d}_vol.npy", vol)
        np.save(Path(data_dir) / f"sub_{i:03d}_roi_ts.npy", roi_ts)
        np.save(Path(data_dir) / f"sub_{i:03d}_conn.npy", conn)
        samples.append({
            "fmri": str(Path(data_dir) / f"sub_{i:03d}_vol.npy"),
            "roi_timeseries": roi_ts,
            "roi_connectivity": conn,
            "label": label,
        })
    return samples
