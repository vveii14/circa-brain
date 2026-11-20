"""Shared helpers: cross-validation splits, model construction, metrics."""

from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from models import ImageROIFusionModel


def make_splits(labels: np.ndarray, n_folds: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Subject-level stratified splits: one (train, val, test) index triple per fold.

    The data are divided into `n_folds` stratified parts. For fold k, part k is the test set,
    part k+1 (cyclically) is the validation set used for model selection, and the remaining
    parts are used for training. Every subject is tested exactly once.
    """
    parts = [test for _, test in StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
             .split(np.zeros(len(labels)), labels)]
    splits = []
    for k in range(n_folds):
        val_k = (k + 1) % n_folds
        train = np.concatenate([parts[j] for j in range(n_folds) if j not in (k, val_k)])
        splits.append((np.sort(train), parts[val_k], parts[k]))
    return splits


def build_model(cfg: dict) -> ImageROIFusionModel:
    return ImageROIFusionModel(
        n_rois=cfg["n_rois"],
        num_classes=cfg["num_classes"],
        use_image_branch=cfg["use_image_branch"],
        use_roi_branch=cfg["use_roi_branch"],
        image_encoder=cfg["image_encoder"],
        image_feat_dim=cfg["image_feat_dim"],
        roi_encoder=cfg["roi_encoder"],
        roi_feat_dim=cfg["roi_feat_dim"],
        fusion=cfg["fusion"],
        contrastive_temperature=cfg.get("contrastive_temperature", 0.07),
        contrastive_weight=cfg.get("contrastive_weight", 0.1),
        dropout=cfg.get("dropout", 0.3),
        image_encoder_kwargs=cfg.get("image_encoder_kwargs"),
        roi_encoder_kwargs=cfg.get("roi_encoder_kwargs"),
        fusion_kwargs=cfg.get("fusion_kwargs"),
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """Binary metrics in %. Predictions are the argmax class; AUC uses the positive-class probability."""
    y_true = np.asarray(y_true).ravel()
    y_pred = prob.argmax(axis=1)
    try:
        auc = roc_auc_score(y_true, prob[:, 1]) * 100.0
    except ValueError:  # only one class present (e.g. tiny demo split)
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100.0,
        "auc": auc,
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100.0,
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0,
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0,
    }


METRIC_KEYS = ("accuracy", "auc", "sensitivity", "specificity", "f1")


def summarize(rows: List[dict]) -> str:
    lines = []
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows], dtype=float)
        std = np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0
        per_fold = ", ".join(f"{v:.2f}" for v in vals)
        lines.append(f"  {k:12s} {np.nanmean(vals):6.2f} +- {std:5.2f}   [{per_fold}]")
    return "\n".join(lines)
