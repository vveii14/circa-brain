"""
ADNI FDG-PET: FDG.csv (IID + 90 ROI columns) + disease label.xlsx.
Labels: drop SMC; merge EMCI/LMCI/MCI -> MCI; final classes CN=0, MCI=1, AD=2.
Used for BrainNet ROI-only (90x90 per-subject matrix from ROI vector).
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_adni_fdg_pet(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Load ADNI FDG-PET ROI data and labels.
    - Drops SMC.
    - Maps EMCI, LMCI, MCI -> 1 (MCI); CN -> 0; AD -> 2.
    Returns:
        roi_matrix: (n, 90) float32 ROI uptake values
        labels: (n,) int 0=CN, 1=MCI, 2=AD
        iids: list of IID strings
        matrices_90x90: (n, 90, 90) for BrainNet, from normalized outer product of ROI vector
    """
    data_dir = Path(data_dir)
    fdg_path = data_dir / "FDG.csv"
    label_path = data_dir / "disease label.xlsx"
    if not fdg_path.exists():
        raise FileNotFoundError(f"FDG.csv not found: {fdg_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"disease label.xlsx not found: {label_path}")

    fdg = pd.read_csv(fdg_path)
    # IID + 90 ROI columns
    roi_cols = [c for c in fdg.columns if c != "IID"]
    n_rois = len(roi_cols)
    if n_rois != 90:
        raise ValueError(f"Expected 90 ROI columns, got {n_rois}")

    xl = pd.ExcelFile(label_path)
    lb = pd.read_excel(xl, sheet_name=0)
    lb = lb.rename(columns={"Subject ID": "IID"})
    lb["IID"] = lb["IID"].astype(str)
    fdg["IID"] = fdg["IID"].astype(str)

    merged = fdg.merge(lb[["IID", "Research Group"]], on="IID", how="inner")
    # Drop SMC
    merged = merged[merged["Research Group"] != "SMC"].copy()

    def _map(rg: str) -> int:
        if rg == "CN":
            return 0
        if rg in ("EMCI", "LMCI", "MCI"):
            return 1
        if rg == "AD":
            return 2
        return -1

    merged["label"] = merged["Research Group"].map(_map)
    merged = merged[merged["label"] >= 0].reset_index(drop=True)

    iids = merged["IID"].tolist()
    roi_matrix = merged[roi_cols].astype(np.float32).values
    labels = merged["label"].astype(np.int32).values

    # (n, 90, 90) for BrainNet: normalized outer product per subject
    n = roi_matrix.shape[0]
    matrices_90x90 = np.zeros((n, n_rois, n_rois), dtype=np.float32)
    for i in range(n):
        v = roi_matrix[i]
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        m = np.mean(v)
        s = np.std(v)
        if s > 1e-8:
            z = (v - m) / s
        else:
            z = v - m
        matrices_90x90[i] = np.outer(z, z)
        np.fill_diagonal(matrices_90x90[i], 1.0)

    return roi_matrix, labels, iids, matrices_90x90


def get_adni_meta(data_dir: str) -> Tuple[List[int], np.ndarray]:
    """Return (indices 0..N-1, labels array) for stratified split."""
    _, labels, _, _ = load_adni_fdg_pet(data_dir)
    return list(range(len(labels))), np.asarray(labels)
