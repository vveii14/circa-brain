"""
Deterministic 5-fold cross-validation for joint imaging-ROI brain-disorder
classification (image-only / ROI-only / fusion: concat | contrastive | cross_attention).

Usage:
  # Fusion (3D-SCTF image + NeuroGraph ROI, contrastive alignment)
  python train_kfold.py --dataset adhd --mode fusion --fusion contrastive \
    --config configs/adhd_3dsctf_neurograph_contrastive.yaml --n_folds 5 --seed 42
  # Image-only / ROI-only
  python train_kfold.py --dataset adhd --mode image_only --config <cfg> --n_folds 5
  python train_kfold.py --dataset adhd --mode roi_only  --config <cfg> --n_folds 5

Reproducibility: with a fixed --seed the full 5-fold run is bit-for-bit
deterministic (see set_seed); rerunning yields identical per-fold metrics.
"""

# Deterministic cuBLAS GEMMs (must be set before the first CUDA context is created).
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

try:  # optional: this clean release ships only the ADHD dataset
    from data.brain_structure import BrainStructureDataset
except Exception:
    BrainStructureDataset = None
from data.adhd import ADHDDataset, get_adhd_meta
from models import ImageROIFusionModel


def set_seed(seed: int) -> None:
    """Seed every RNG and force deterministic kernels for bit-reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: a few 3D ops lack deterministic CUDA kernels; warn instead of crashing.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """Per-worker RNG seeding so DataLoader shuffling is reproducible (num_workers>0)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate(batch):
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    out = {"label": labels}
    if "image" in batch[0]:
        out["image"] = torch.stack([b["image"] for b in batch])
    if "roi_data" in batch[0]:
        out["roi_data"] = torch.stack([b["roi_data"] for b in batch])
    return out


def compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int):
    """Compute Accuracy (%), AUC (%), Sensitivity (%), Specificity (%), F1-score (%)."""
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "auc": 0.0, "sensitivity": 0.0, "specificity": 0.0, "f1": 0.0}
    acc = accuracy_score(y_true, y_pred) * 100.0
    # AUC: binary use prob of positive class; multi-class use ovr
    if num_classes == 2:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            prob_pos = y_prob[:, 1]
        else:
            prob_pos = y_prob.ravel()
        try:
            auc = roc_auc_score(y_true, prob_pos) * 100.0
        except ValueError:
            auc = 0.0
    else:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") * 100.0
        except ValueError:
            auc = 0.0
    # Binary: sensitivity = recall class 1, specificity = recall class 0
    if num_classes == 2:
        try:
            sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100.0
        except Exception:
            sensitivity = 0.0
        try:
            specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0
        except Exception:
            specificity = 0.0
    else:
        # Multi-class: sensitivity = macro-avg per-class recall (TP / (TP+FN))
        #              specificity = macro-avg per-class TN / (TN+FP) via one-vs-rest
        try:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            sens_list, spec_list = [], []
            for c in range(num_classes):
                tp = cm[c, c]
                fn = cm[c, :].sum() - tp
                fp = cm[:, c].sum() - tp
                tn = cm.sum() - tp - fn - fp
                sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            sensitivity = float(np.mean(sens_list)) * 100.0
            specificity = float(np.mean(spec_list)) * 100.0
        except Exception:
            sensitivity = 0.0
            specificity = 0.0
    # F1: macro
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    return {
        "accuracy": acc,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
    }


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("lr", "weight_decay", "contrastive_temperature", "contrastive_weight", "dropout"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])
    for k in ("epochs", "batch_size", "n_rois", "num_classes", "seed"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])
    return cfg


def run_one_fold(cfg, data_path, train_idx, val_idx, fold, device, log_fn, image_shape, dataset_name="brain_structure", ckpt_path=None, test_idx=None):
    use_image = cfg["use_image_branch"]
    use_roi = cfg["use_roi_branch"]
    log_fn(f"[Fold {fold}] Step: loading data (dataset={dataset_name}, path={data_path})")
    if dataset_name in ("adhd", "abide"):
        use_test_set_only = cfg.get("adhd_use_test_set_only", False)
        roi_matrices_file = "test_roi_matrices_775.npy" if use_test_set_only else cfg.get("roi_matrices_file")
        fold_prefix = cfg.get("_fold_prefix", "adhd")
        fold_suffix = cfg.get("_fold_suffix", "775")
        train_ds = ADHDDataset(
            data_dir=data_path,
            use_image=use_image,
            use_roi=use_roi,
            n_rois=cfg["n_rois"],
            indices=train_idx,
            roi_matrices_file=roi_matrices_file,
            use_test_set_only=use_test_set_only,
            fold_prefix=fold_prefix,
            fold_suffix=fold_suffix,
        )
        val_ds = ADHDDataset(
            data_dir=data_path,
            use_image=use_image,
            use_roi=use_roi,
            n_rois=cfg["n_rois"],
            indices=val_idx,
            roi_matrices_file=roi_matrices_file,
            use_test_set_only=use_test_set_only,
            fold_prefix=fold_prefix,
            fold_suffix=fold_suffix,
        )
        test_ds = None
        if test_idx is not None:
            test_ds = ADHDDataset(
                data_dir=data_path,
                use_image=use_image,
                use_roi=use_roi,
                n_rois=cfg["n_rois"],
                indices=test_idx,
                roi_matrices_file=roi_matrices_file,
                use_test_set_only=use_test_set_only,
                fold_prefix=fold_prefix,
                fold_suffix=fold_suffix,
            )
    else:
        train_ds = BrainStructureDataset(
            cache_dir=data_path,
            split="train",
            use_image=use_image,
            use_roi=use_roi,
            target_shape=tuple(image_shape),
            indices=train_idx,
        )
        val_ds = BrainStructureDataset(
            cache_dir=data_path,
            split="train",
            use_image=use_image,
            use_roi=use_roi,
            target_shape=tuple(image_shape),
            indices=val_idx,
        )
        test_ds = None
    n_train, n_val = len(train_ds), len(val_ds)
    n_test = len(test_ds) if test_ds is not None else 0
    log_fn(f"[Fold {fold}] Data loaded: train n={n_train}, val n={n_val}" + (f", test n={n_test}" if n_test else ""))

    # Compute per-fold class weights to handle class imbalance
    train_labels_np = np.array([train_ds[i]["label"] for i in range(len(train_ds))])
    _classes = np.arange(cfg["num_classes"])
    _cw = compute_class_weight("balanced", classes=_classes, y=train_labels_np)
    class_weight_tensor = torch.tensor(_cw, dtype=torch.float32)
    log_fn(f"[Fold {fold}] Class weights (balanced): {dict(zip(_classes.tolist(), [round(float(w), 3) for w in _cw]))}")

    # Seed the shuffler so the per-fold batch order is reproducible across runs.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(cfg["seed"]) + fold)
    # drop_last=True to avoid batch_size=1 causing BatchNorm to fail
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
        generator=loader_gen,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,  # Never drop samples during validation/evaluation
    )
    n_batches_train = len(train_loader)
    log_fn(f"[Fold {fold}] DataLoader: train_batches={n_batches_train}, val_batches={len(val_loader)}")

    log_fn(f"[Fold {fold}] Step: building model (image_enc={cfg.get('image_encoder')}, roi_enc={cfg.get('roi_encoder')}, fusion={cfg.get('fusion')})")
    if cfg.get("roi_encoder") == "neurograph":
        roi_input = "precomputed roi_matrices" if cfg.get("roi_matrices_file") else "roi vectors (C computed in encoder)"
        log_fn(f"[Fold {fold}] NeuroGraph ROI encoder: n_rois={cfg['n_rois']}, input={roi_input}")
    image_encoder_kwargs = dict(cfg.get("image_encoder_kwargs") or {})
    if "img_size" not in image_encoder_kwargs and image_shape is not None:
        image_encoder_kwargs["img_size"] = tuple(image_shape)
    roi_encoder_kwargs = dict(cfg.get("roi_encoder_kwargs") or {})
    model = ImageROIFusionModel(
        n_rois=cfg["n_rois"],
        num_classes=cfg["num_classes"],
        use_image_branch=use_image,
        use_roi_branch=use_roi,
        image_encoder=cfg.get("image_encoder", "conv3d"),
        image_feat_dim=cfg["image_feat_dim"],
        roi_encoder=cfg.get("roi_encoder", "roi_vector"),
        roi_feat_dim=cfg["roi_feat_dim"],
        fusion=cfg["fusion"],
        contrastive_temperature=cfg.get("contrastive_temperature", 0.07),
        contrastive_weight=cfg.get("contrastive_weight", 0.1),
        dropout=cfg.get("dropout", 0.3),
        image_encoder_kwargs=image_encoder_kwargs,
        roi_encoder_kwargs=roi_encoder_kwargs,
        fusion_kwargs=cfg.get("fusion_kwargs"),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"[Fold {fold}] Model built: {n_params} parameters, device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    log_fn(f"[Fold {fold}] Optimizer: AdamW lr={cfg['lr']} weight_decay={cfg.get('weight_decay', 1e-4)}")
    # Cosine annealing LR scheduler helps image models (ViT) converge
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    log_fn(f"[Fold {fold}] Scheduler: CosineAnnealingLR T_max={cfg['epochs']}")
    # Optional regularization (config-driven): label smoothing on CE, gradient-norm clipping.
    # Defaults (0.0 / none) preserve the original behaviour when the keys are absent.
    label_smoothing = float(cfg.get("label_smoothing", 0.0) or 0.0)
    max_grad_norm = cfg.get("max_grad_norm", None)
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
    log_fn(f"[Fold {fold}] Regularization: label_smoothing={label_smoothing:.3f} max_grad_norm={max_grad_norm if max_grad_norm is not None else 'none'}")
    log_fn(f"[Fold {fold}] Step: training ({cfg['epochs']} epochs)")
    best_val_auc = -1.0
    best_state = None
    # Use larger patience for image/fusion models that need warmup (default 30, image/fusion 50)
    _default_patience = 50 if use_image else 30
    early_stop_patience = cfg.get("early_stop_patience", _default_patience)
    patience_counter = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
            image = batch.get("image")
            roi_data = batch.get("roi_data")
            labels = batch["label"].to(device)
            if image is not None:
                image = image.to(device)
            if roi_data is not None:
                roi_data = roi_data.to(device)
            opt.zero_grad()
            # Single forward: reuse the embeddings that feed the classifier for the
            # contrastive term. This halves encoder compute vs a second forward and,
            # crucially, aligns the *same* dropout/drop-path sample the classifier sees
            # (a second forward would re-sample, aligning embeddings the head never uses).
            use_contrastive = use_image and use_roi and cfg.get("fusion") == "contrastive"
            if use_contrastive:
                logits, (z_img, z_roi) = model(image=image, roi_data=roi_data, return_embeddings=True)
            else:
                logits = model(image=image, roi_data=roi_data)
            loss = F.cross_entropy(logits, labels, weight=class_weight_tensor.to(device), label_smoothing=label_smoothing)
            if use_contrastive:
                loss = loss + cfg.get("contrastive_weight", 0.1) * model.fusion_head.get_contrastive_loss(z_img, z_roi)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_n += labels.size(0)
            # Log GPU memory once per fold (first step of epoch 0)
            if epoch == 0 and train_n == labels.size(0) and torch.cuda.is_available():
                alloc_gib = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserv_gib = torch.cuda.memory_reserved(device) / (1024 ** 3)
                log_fn(f"[Fold {fold}] GPU memory after 1st step: allocated {alloc_gib:.2f} GiB, reserved {reserv_gib:.2f} GiB (batch_size={labels.size(0)})")
        scheduler.step()
        train_loss = train_loss_sum / train_n if train_n else 0.0

        # Validation: compute AUC for best-model selection (better than acc for imbalanced multi-class)
        model.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                image = batch.get("image")
                roi_data = batch.get("roi_data")
                labels = batch["label"].to(device)
                if image is not None:
                    image = image.to(device)
                if roi_data is not None:
                    roi_data = roi_data.to(device)
                logits = model(image=image, roi_data=roi_data)
                val_logits_list.append(logits.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())
        val_logits_np = np.concatenate(val_logits_list, axis=0)
        val_labels_np = np.concatenate(val_labels_list, axis=0)
        # Numerically stable softmax
        val_logits_shifted = val_logits_np - val_logits_np.max(axis=1, keepdims=True)
        val_probs = np.exp(val_logits_shifted) / np.exp(val_logits_shifted).sum(axis=1, keepdims=True)
        val_preds = np.argmax(val_logits_np, axis=1)
        val_acc = accuracy_score(val_labels_np, val_preds)
        num_classes_val = cfg["num_classes"]
        try:
            if num_classes_val == 2:
                val_auc = roc_auc_score(val_labels_np, val_probs[:, 1])
            else:
                val_auc = roc_auc_score(val_labels_np, val_probs, multi_class="ovr", average="macro")
        except ValueError:
            val_auc = 0.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": best_state, "val_auc": best_val_auc, "epoch": epoch, "fold": fold, "cfg": cfg}, ckpt_path)
        else:
            patience_counter += 1
        cur_lr = scheduler.get_last_lr()[0]
        log_fn(f"  [Fold {fold}] epoch {epoch+1}/{cfg['epochs']} train_loss {train_loss:.4f} val_acc {val_acc:.4f} val_auc {val_auc:.4f} best_val_auc {best_val_auc:.4f} lr {cur_lr:.2e}")
        if patience_counter >= early_stop_patience:
            log_fn(f"  [Fold {fold}] Early stopping at epoch {epoch+1} (patience={early_stop_patience})")
            break

    # ---------- Test metrics on held-out fold (with best model) ----------
    log_fn(f"[Fold {fold}] Step: evaluating on val set (best model)")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            image = batch.get("image")
            roi_data = batch.get("roi_data")
            labels = batch["label"].to(device)
            if image is not None:
                image = image.to(device)
            if roi_data is not None:
                roi_data = roi_data.to(device)
            logits = model(image=image, roi_data=roi_data)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_logits_shifted = all_logits - all_logits.max(axis=1, keepdims=True)
    y_prob = np.exp(all_logits_shifted) / np.exp(all_logits_shifted).sum(axis=1, keepdims=True)
    y_pred = np.argmax(all_logits, axis=1)
    num_classes = cfg["num_classes"]
    metrics = compute_test_metrics(all_labels, y_pred, y_prob, num_classes)
    log_fn(f"[Fold {fold}] Val metrics:")
    log_fn(f"  Accuracy (%)     = {metrics['accuracy']:.2f}")
    log_fn(f"  AUC (%)         = {metrics['auc']:.2f}")
    log_fn(f"  F1-score (%)    = {metrics['f1']:.2f}")
    log_fn(f"[Fold {fold}] Step: done. best_val_auc={best_val_auc:.4f} (best AUC on val during training)")

    # Optional: evaluate on fixed test set (6/2/2 when use_test_set_only with 10 samples)
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, collate_fn=collate, drop_last=False)
        all_logits_t, all_labels_t = [], []
        with torch.no_grad():
            for batch in test_loader:
                image = batch.get("image")
                roi_data = batch.get("roi_data")
                labels = batch["label"].to(device)
                if image is not None:
                    image = image.to(device)
                if roi_data is not None:
                    roi_data = roi_data.to(device)
                logits = model(image=image, roi_data=roi_data)
                all_logits_t.append(logits.cpu().numpy())
                all_labels_t.append(labels.cpu().numpy())
        all_logits_t = np.concatenate(all_logits_t, axis=0)
        all_labels_t = np.concatenate(all_labels_t, axis=0)
        all_logits_t_shifted = all_logits_t - all_logits_t.max(axis=1, keepdims=True)
        y_prob_t = np.exp(all_logits_t_shifted) / np.exp(all_logits_t_shifted).sum(axis=1, keepdims=True)
        y_pred_t = np.argmax(all_logits_t, axis=1)
        test_metrics = compute_test_metrics(all_labels_t, y_pred_t, y_prob_t, num_classes)
        log_fn(f"[Fold {fold}] Held-out test set (n={len(test_ds)}) metrics:")
        log_fn(f"  Accuracy (%)     = {test_metrics['accuracy']:.2f}")
        log_fn(f"  AUC (%)         = {test_metrics['auc']:.2f}")
        log_fn(f"  F1-score (%)    = {test_metrics['f1']:.2f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/brain_structure.yaml")
    parser.add_argument("--dataset", type=str, default="brain_structure", choices=["brain_structure", "adhd", "abide"])
    parser.add_argument("--mode", type=str, choices=["image_only", "roi_only", "fusion"], default="image_only")
    parser.add_argument("--fusion", type=str, choices=["concat", "contrastive", "cross_attention"], default=None, help="Fusion method (only used when --mode fusion)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_folds", type=int, default=None, help="Run only first K folds (smoke test).")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs in config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size in config.")
    parser.add_argument("--lr", type=float, default=None, help="Override lr in config.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight_decay in config.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance_classes", action="store_true", help="For brain_structure: undersample to equalize AD/CN counts before 5-fold split.")
    parser.add_argument("--use_test_set", action="store_true", help="Demo mode: run on the bundled 10-sample set (adhd_test_775.npy) instead of the full folds.")
    parser.add_argument("--output_dir", type=str, default=None, help="If set, write log and checkpoints under this dir (e.g. for parallel runs).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.fusion is not None:
        cfg["fusion"] = args.fusion
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if args.weight_decay is not None:
        cfg["weight_decay"] = float(args.weight_decay)
    if not cfg.get("data_root") or cfg.get("data_root") == ".":
        cfg["data_root"] = "."
    cfg["seed"] = args.seed
    if args.use_test_set:
        cfg["adhd_use_test_set_only"] = True  # demo: 10-sample bundled set
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = cfg.get("image_shape", [96, 96, 96])

    if args.mode == "image_only":
        cfg["use_image_branch"] = True
        cfg["use_roi_branch"] = False
    elif args.mode == "roi_only":
        cfg["use_image_branch"] = False
        cfg["use_roi_branch"] = True
        cfg.setdefault("roi_encoder", "neurograph")
    else:
        cfg["use_image_branch"] = True
        cfg["use_roi_branch"] = True
        cfg["roi_encoder"] = cfg.get("roi_encoder", "neurograph")

    balanced_indices = None  # used only for brain_structure when --balance_classes
    if args.dataset in ("adhd", "abide"):
        data_path = cfg.get("adhd_data_dir", str(Path(cfg["data_root"]) / "adhd"))
        fold_prefix = "adhd"
        fold_suffix = "775"
        if args.dataset == "abide":
            data_path = cfg.get("abide_data_dir", str(Path(cfg["data_root"]) / "abide"))
            meta_path = Path(data_path) / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta_cfg = json.load(f)
                fold_prefix = str(meta_cfg.get("fold_prefix", "abide"))
                fold_suffix = str(meta_cfg.get("fold_suffix", "987"))
            else:
                fold_prefix = "abide"
                fold_suffix = "987"
            cfg["abide_data_dir"] = data_path
            cfg["roi_matrices_file"] = cfg.get("roi_matrices_file", f"roi_matrices_{fold_suffix}.npy")
            cfg["roi_vectors_file"] = cfg.get("roi_vectors_file", f"roi_vectors_{fold_suffix}.npy")
        cfg["_fold_prefix"] = fold_prefix
        cfg["_fold_suffix"] = fold_suffix
        indices, labels = get_adhd_meta(data_path, fold_prefix=fold_prefix, fold_suffix=fold_suffix,
                                        use_test_set_only=cfg.get("adhd_use_test_set_only", False))
        n = len(indices)
        labels = np.asarray(labels)
        labels_for_skf = labels
    else:
        data_path = cfg.get("brain_structure_cache_dir", str(Path(cfg["data_root"]) / "brain_structure_cache"))
        meta_path = Path(data_path) / "train" / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}. Run preprocess first.")
        with open(meta_path) as f:
            meta = json.load(f)
        labels = np.array([m["label"] for m in meta])
        n_total = len(meta)
        if args.balance_classes:
            # Undersample so AD and CN counts are equal (random sample from majority).
            idx_by_class = [np.where(labels == c)[0] for c in range(cfg["num_classes"])]
            n_per_class = [len(idx_by_class[c]) for c in range(cfg["num_classes"])]
            n_min = int(min(n_per_class))
            rng = np.random.default_rng(cfg["seed"])
            balanced_indices = np.concatenate([
                rng.choice(idx_by_class[c], size=n_min, replace=False) for c in range(cfg["num_classes"])
            ]).astype(int)
            rng.shuffle(balanced_indices)
            balanced_indices = balanced_indices.tolist()
            n = len(balanced_indices)
            labels_for_skf = labels[balanced_indices]
        else:
            balanced_indices = None
            n = n_total
            labels_for_skf = labels
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=cfg["seed"])
    fold_accs = []
    log_suffix = f"{args.mode}_{cfg['fusion']}" if args.mode == "fusion" and args.fusion else args.mode
    if args.dataset == "adhd":
        log_suffix = f"adhd_{log_suffix}"
    if args.output_dir:
        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        log_path = out_root / "train.log"
        cfg["_checkpoint_dir"] = str(out_root / "checkpoints")
    else:
        log_path = Path(cfg.get("data_root", ".")) / f"train_kfold_{log_suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(s):
        with open(log_path, "a") as f:
            f.write(s + "\n")
        print(s, flush=True)

    # ---------- Log: run summary ----------
    log("=" * 60)
    log("5-fold cross-validation")
    log("  Note: Each fold trains on 4 parts and evaluates on the held-out part; that held-out accuracy is the 'val_acc' (i.e. test acc for that fold). No separate test set.")
    log("=" * 60)
    log(f"[Data] dataset={args.dataset} data_path={data_path}")
    log(f"[Data] n_samples={n} n_folds={args.n_folds} seed={cfg['seed']}")
    unique, counts = np.unique(labels_for_skf, return_counts=True)
    log(f"[Data] label_distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    if args.dataset == "brain_structure" and args.balance_classes:
        log("[Data] balance_classes=True: AD/CN undersampled to equal counts.")
    log("-")
    log("[Config] (main run params)")
    log(f"  config_file={args.config}")
    log(f"  mode={args.mode} fusion={cfg.get('fusion')}")
    log(f"  use_image_branch={cfg['use_image_branch']} use_roi_branch={cfg['use_roi_branch']}")
    log(f"  image_encoder={cfg.get('image_encoder')} roi_encoder={cfg.get('roi_encoder')}")
    log(f"  image_feat_dim={cfg.get('image_feat_dim')} roi_feat_dim={cfg.get('roi_feat_dim')}")
    log(f"  batch_size={cfg['batch_size']} epochs={cfg['epochs']} lr={cfg['lr']} weight_decay={cfg.get('weight_decay')}")
    log(f"  num_classes={cfg['num_classes']} n_rois={cfg['n_rois']} dropout={cfg.get('dropout')}")
    if cfg.get("image_encoder_kwargs"):
        log(f"  image_encoder_kwargs={cfg['image_encoder_kwargs']}")
    log(f"  device={device} image_shape={list(image_shape)}")
    log("-")
    log(f"=== Start 5-fold CV (dataset={args.dataset} mode={args.mode} n_folds={args.n_folds} n_samples={n}) ===")
    fold_metrics_list = []
    for fold, (train_pos, val_pos) in enumerate(skf.split(np.arange(n), labels_for_skf)):
        if args.max_folds is not None and (fold + 1) > int(args.max_folds):
            log(f"[INFO] max_folds={args.max_folds} reached. Stopping early for smoke test.")
            break
        if balanced_indices is not None:
            train_idx = [balanced_indices[i] for i in train_pos]
            val_idx = [balanced_indices[i] for i in val_pos]
        else:
            train_idx = train_pos.tolist()
            val_idx = val_pos.tolist()
        log(f"--- Fold {fold+1}/{args.n_folds} (train_n={len(train_idx)} val_n={len(val_idx)}) ---")
        if cfg.get("_checkpoint_dir"):
            ckpt_dir = Path(cfg["_checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"fold{fold+1}_best.pth"
        else:
            ckpt_path = Path(cfg.get("data_root", ".")) / "checkpoints" / log_suffix / f"fold{fold+1}_best.pth"
        metrics = run_one_fold(cfg, data_path, train_idx, val_idx, fold + 1, device, log, image_shape, dataset_name=args.dataset, ckpt_path=str(ckpt_path))
        fold_metrics_list.append(metrics)
        log(f"Fold {fold+1} done. Accuracy={metrics['accuracy']:.2f}% AUC={metrics['auc']:.2f}% F1={metrics['f1']:.2f}%")

    # ---------- 5-fold summary: mean ± std for each test metric ----------
    if not fold_metrics_list:
        log("[WARNING] No folds were completed; nothing to summarize.")
        return
    keys = ["accuracy", "auc", "f1"]
    means = {k: float(np.mean([m[k] for m in fold_metrics_list])) for k in keys}
    # ddof=1 for unbiased std estimate across k folds
    stds = {k: float(np.std([m[k] for m in fold_metrics_list], ddof=1)) for k in keys}
    log("=" * 60)
    log("5-fold CV done. Test metrics (held-out) summary:")
    log("  Metric          | Mean ± Std (%)  | Per-fold")
    log("  ----------------|-----------------|------------------------")
    log(f"  Accuracy (%)    | {means['accuracy']:.2f} ± {stds['accuracy']:.2f}   | {[round(m['accuracy'], 2) for m in fold_metrics_list]}")
    log(f"  AUC (%)         | {means['auc']:.2f} ± {stds['auc']:.2f}   | {[round(m['auc'], 2) for m in fold_metrics_list]}")
    log(f"  F1-score (%)    | {means['f1']:.2f} ± {stds['f1']:.2f}   | {[round(m['f1'], 2) for m in fold_metrics_list]}")
    log("=" * 60)
    print(f"Done. {args.mode}: Accuracy {means['accuracy']:.2f}±{stds['accuracy']:.2f}%  AUC {means['auc']:.2f}±{stds['auc']:.2f}%  F1 {means['f1']:.2f}±{stds['f1']:.2f}%")


if __name__ == "__main__":
    main()
