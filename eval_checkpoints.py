"""Run saved checkpoints with a fixed classification threshold.
Usage: python eval_checkpoints.py [CHECKPOINT_DIR] [DATA_DIR]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))

from data.adhd import ADHDDataset, get_adhd_meta  # noqa: E402
from models import ImageROIFusionModel  # noqa: E402
from inference_utils import (  # noqa: E402
    collate,
    compute_test_metrics,
)

CKPT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "checkpoints"
DATA = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "data" / "adhd")
KEYS = ["accuracy", "auc", "sensitivity", "specificity", "f1"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


indices, labels = get_adhd_meta(DATA)
labels = np.asarray(labels)
splits = list(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(indices)), labels)
)


tuned_all, argmax_all = [], []

for fold, (_, val_idx) in enumerate(splits, 1):
    ckpt = torch.load(CKPT_DIR / f"fold{fold}_best.pth", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = ImageROIFusionModel(
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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = ADHDDataset(
        data_dir=DATA,
        use_image=True,
        use_roi=True,
        n_rois=cfg["n_rois"],
        indices=val_idx.tolist(),
        roi_matrices_file=cfg.get("roi_matrices_file", "roi_matrices_775.npy"),
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, collate_fn=collate)

    logits, y_true = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(image=batch["image"].to(device), roi_data=batch["roi_data"].to(device))
            logits.append(out.cpu().numpy())
            y_true.append(batch["label"].numpy())
    logits, y_true = np.concatenate(logits), np.concatenate(y_true)

    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    argmax_pred = logits.argmax(axis=1)
    argmax_metrics = compute_test_metrics(y_true, argmax_pred, prob, 2)

    # Apply a fixed threshold; no threshold fitting during evaluation.
    thr = 0.5
    tuned_metrics = compute_test_metrics(y_true, (prob[:, 1] >= thr).astype(np.int64), prob, 2)
    tuned_all.append(tuned_metrics)
    argmax_all.append(argmax_metrics)

    print(
        f"fold{fold}: "
        f"threshold={thr:.4f} test_metrics={tuned_metrics}", flush=True,
    )

for name, rows in [("fixed threshold 0.5", tuned_all), ("argmax", argmax_all)]:
    print(f"=== mean +- std, {name} ===")
    for k in KEYS:
        values = [row[k] for row in rows]
        print(f"  {k:12s} {np.mean(values):.2f} +- {np.std(values, ddof=1):.2f}")
