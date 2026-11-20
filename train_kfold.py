"""
5-fold cross-validation training for joint imaging + ROI-graph classification.

    # Proposed model: 3DSC-TF (imaging) + NeuroGraph (ROI) with cross-view contrastive alignment
    python train_kfold.py --config config.yaml --data_dir data/adhd --output_dir runs/adhd_contrastive

    # Ablations with the same backbones
    python train_kfold.py --config config.yaml --fusion concat     --output_dir runs/adhd_concat
    python train_kfold.py --config config.yaml --mode image_only   --output_dir runs/adhd_image_only
    python train_kfold.py --config config.yaml --mode roi_only     --output_dir runs/adhd_roi_only

    # Smoke test on the bundled synthetic samples
    python train_kfold.py --config config.yaml --demo --epochs 2 --batch_size 4 --output_dir runs/demo

Protocol (utils.make_splits): the subjects are split into 5 stratified parts. In fold k, part k is
the test set, part k+1 is the validation set and the other three parts are the training set. The
checkpoint with the best validation AUC is kept, and only that checkpoint is evaluated on the
test set. Reported numbers are the mean +- std of the five test-set results.
"""

import os

# Deterministic cuBLAS GEMMs (must be set before the first CUDA context is created).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from data import ADHDDataset, collate, get_adhd_meta
from utils import build_model, compute_metrics, make_splits, softmax, summarize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # A few 3D ops have no deterministic CUDA kernel; warn instead of failing.
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        image = batch["image"].to(device) if "image" in batch else None
        roi = batch["roi_data"].to(device) if "roi_data" in batch else None
        probs.append(softmax(model(image=image, roi_data=roi).float().cpu().numpy()))
        labels.append(batch["label"].numpy())
    return np.concatenate(labels), np.concatenate(probs)


def run_fold(cfg, args, fold, train_idx, val_idx, test_idx, device, log, ckpt_path):
    ds_kw = dict(use_image=cfg["use_image_branch"], use_roi=cfg["use_roi_branch"], n_rois=cfg["n_rois"],
                 prefix=args.prefix, demo=args.demo, roi_matrices_file=cfg.get("roi_matrices_file"))
    train_ds = ADHDDataset(args.data_dir, indices=train_idx.tolist(), **ds_kw)
    val_ds = ADHDDataset(args.data_dir, indices=val_idx.tolist(), **ds_kw)
    test_ds = ADHDDataset(args.data_dir, indices=test_idx.tolist(), **ds_kw)
    log(f"[fold {fold}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    gen = torch.Generator()
    gen.manual_seed(cfg["seed"] + fold)
    # drop_last avoids a final batch of size 1, which BatchNorm cannot handle in training mode.
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                              num_workers=0, collate_fn=collate, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, collate_fn=collate)

    train_labels = np.array([int(train_ds._data[i]["label"]) for i in train_ds.indices])
    class_weight = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(cfg["num_classes"]), y=train_labels),
        dtype=torch.float32, device=device,
    )

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    use_contrastive = cfg["use_image_branch"] and cfg["use_roi_branch"] and cfg["fusion"] == "contrastive"
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    max_grad_norm = cfg.get("max_grad_norm")

    best_auc, best_epoch, best_state = -1.0, -1, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for batch in train_loader:
            image = batch["image"].to(device) if "image" in batch else None
            roi = batch["roi_data"].to(device) if "roi_data" in batch else None
            labels = batch["label"].to(device)
            opt.zero_grad()
            if use_contrastive:
                # The contrastive term uses the same embeddings (same dropout sample) as the classifier.
                logits, (z_img, z_roi) = model(image=image, roi_data=roi, return_embeddings=True)
            else:
                logits = model(image=image, roi_data=roi)
            loss = F.cross_entropy(logits, labels, weight=class_weight, label_smoothing=label_smoothing)
            if use_contrastive:
                loss = loss + cfg["contrastive_weight"] * model.fusion_head.get_contrastive_loss(z_img, z_roi)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            opt.step()
            loss_sum += loss.item() * labels.size(0)
            n_seen += labels.size(0)
        scheduler.step()

        y_val, p_val = predict(model, val_loader, device)
        val_auc = compute_metrics(y_val, p_val)["auc"]
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
        log(f"  [fold {fold}] epoch {epoch:3d}/{cfg['epochs']} loss {loss_sum / max(n_seen, 1):.4f} "
            f"val_auc {val_auc:.2f} best {best_auc:.2f}@{best_epoch}")

    model.load_state_dict(best_state)
    y_test, p_test = predict(model, test_loader, device)
    metrics = compute_metrics(y_test, p_test)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)  # weights only; the architecture comes from the config
    log(f"[fold {fold}] best epoch {best_epoch} (val AUC {best_auc:.2f}) -> test: "
        + "  ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default=None, help="overrides data_dir in the config")
    ap.add_argument("--prefix", default="adhd", help="fold-file prefix: adhd or abide")
    ap.add_argument("--mode", choices=["fusion", "image_only", "roi_only"], default="fusion")
    ap.add_argument("--fusion", choices=["contrastive", "concat"], default=None, help="overrides fusion in the config")
    ap.add_argument("--demo", action="store_true", help="use the bundled synthetic samples (smoke test only)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_folds", type=int, default=None, help="run only the first K folds")
    ap.add_argument("--output_dir", default="runs/default")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ("epochs", "batch_size", "seed", "fusion"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["use_image_branch"] = args.mode in ("fusion", "image_only")
    cfg["use_roi_branch"] = args.mode in ("fusion", "roi_only")
    args.data_dir = args.data_dir or cfg.get("data_dir", "data/adhd")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    _, labels = get_adhd_meta(args.data_dir, args.prefix, args.demo)
    splits = make_splits(labels, n_folds=5, seed=cfg["seed"])
    log(f"data_dir={args.data_dir} prefix={args.prefix} demo={args.demo} n={len(labels)} "
        f"labels={dict(zip(*map(np.ndarray.tolist, np.unique(labels, return_counts=True))))}")
    log(f"mode={args.mode} fusion={cfg['fusion']} seed={cfg['seed']} device={device}")
    log("config: " + json.dumps(cfg, sort_keys=True))

    rows = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(splits, 1):
        if args.max_folds is not None and fold > args.max_folds:
            break
        rows.append(run_fold(cfg, args, fold, train_idx, val_idx, test_idx, device, log,
                             out_dir / "checkpoints" / f"fold{fold}_best.pth"))

    log("=== test-set results, mean +- std over folds (%) ===")
    log(summarize(rows))
    log_file.close()


if __name__ == "__main__":
    main()
