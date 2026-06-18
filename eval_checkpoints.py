"""Evaluate the saved paper checkpoints on their seed-42 val folds.
Confirms the 69.29 result is real and recoverable from the trained weights."""
import sys, numpy as np, torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from data.adhd import ADHDDataset, get_adhd_meta
from models import ImageROIFusionModel
from train_kfold import collate, compute_test_metrics

HERE = Path(__file__).parent
CKPT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "checkpoints"
DATA = str(sys.argv[2]) if len(sys.argv) > 2 else str(HERE / "data" / "adhd")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

indices, labels = get_adhd_meta(DATA, fold_prefix="adhd", fold_suffix="775")
labels = np.asarray(labels)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(np.arange(len(indices)), labels))

allm = []
for fold, (tr, va) in enumerate(splits, 1):
    ck = torch.load(CKPT_DIR / f"fold{fold}_best.pth", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if cfg.get("roi_encoder") == "brainnet":  # release renamed this key -> same class
        cfg["roi_encoder"] = "neurograph"
    model = ImageROIFusionModel(
        n_rois=cfg["n_rois"], num_classes=cfg["num_classes"],
        use_image_branch=cfg["use_image_branch"], use_roi_branch=cfg["use_roi_branch"],
        image_encoder=cfg["image_encoder"], image_feat_dim=cfg["image_feat_dim"],
        roi_encoder=cfg["roi_encoder"], roi_feat_dim=cfg["roi_feat_dim"],
        fusion=cfg["fusion"],
        contrastive_temperature=cfg.get("contrastive_temperature", 0.07),
        contrastive_weight=cfg.get("contrastive_weight", 0.1),
        dropout=cfg.get("dropout", 0.3),
        image_encoder_kwargs=cfg.get("image_encoder_kwargs"),
        roi_encoder_kwargs=cfg.get("roi_encoder_kwargs"),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    ds = ADHDDataset(data_dir=DATA, use_image=True, use_roi=True, n_rois=cfg["n_rois"],
                     indices=va.tolist(), roi_matrices_file="roi_matrices_775.npy",
                     fold_prefix="adhd", fold_suffix="775")
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, collate_fn=collate)
    logs, labs = [], []
    with torch.no_grad():
        for b in dl:
            img = b["image"].to(device); roi = b["roi_data"].to(device)
            out = model(image=img, roi_data=roi)
            logs.append(out.cpu().numpy()); labs.append(b["label"].numpy())
    L = np.concatenate(logs); Y = np.concatenate(labs)
    P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    m = compute_test_metrics(Y, L.argmax(1), P, 2)
    allm.append(m)
    print(f"fold{fold}: ckpt_val_auc={ck['val_auc']:.4f} ep={ck['epoch']} | "
          f"Acc={m['accuracy']:.2f} AUC={m['auc']:.2f} F1={m['f1']:.2f}", flush=True)

print("=== MEAN over 5 folds ===")
for k in ["accuracy", "auc", "f1"]:
    print(f"  {k}: {np.mean([m[k] for m in allm]):.2f} +- {np.std([m[k] for m in allm], ddof=1):.2f}")
