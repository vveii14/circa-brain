"""
Evaluate the released checkpoints with 5-fold cross-validation.

    python eval_checkpoints.py                         # ./checkpoints on the full data in ./data/adhd
    python eval_checkpoints.py --data_dir /path/to/adhd
    python eval_checkpoints.py --demo                  # smoke test on the bundled synthetic samples

The checkpoints contain only model weights; the architecture is read from config.yaml.
checkpoints/fold{k}_best.pth is evaluated on the test part of fold k of the stratified
5-fold split (seed 42, see utils.make_splits).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import ADHDDataset, collate, get_adhd_meta
from utils import build_model, compute_metrics, make_splits, softmax, summarize

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def predict(model, loader, device):
    probs, labels = [], []
    for batch in loader:
        logits = model(image=batch["image"].to(device), roi_data=batch["roi_data"].to(device))
        probs.append(softmax(logits.float().cpu().numpy()))
        labels.append(batch["label"].numpy())
    return np.concatenate(labels), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", default=str(HERE / "checkpoints"))
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--data_dir", default=str(HERE / "data" / "adhd"))
    ap.add_argument("--prefix", default="adhd", help="fold-file prefix: adhd or abide")
    ap.add_argument("--demo", action="store_true", help="run every checkpoint on the bundled synthetic samples")
    ap.add_argument("--seed", type=int, default=42, help="seed of the cross-validation split")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["use_image_branch"] = cfg["use_roi_branch"] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, labels = get_adhd_meta(args.data_dir, args.prefix, args.demo)
    if args.demo:
        # Synthetic random samples: this only checks that the pipeline runs.
        test_sets = [np.arange(len(labels))] * 5
    else:
        test_sets = [test for _, _, test in make_splits(labels, n_folds=5, seed=args.seed)]

    rows = []
    for fold, test_idx in enumerate(test_sets, 1):
        state = torch.load(Path(args.ckpt_dir) / f"fold{fold}_best.pth", map_location="cpu", weights_only=True)
        model = build_model(cfg).to(device)
        model.load_state_dict(state)
        model.eval()

        ds = ADHDDataset(args.data_dir, indices=test_idx.tolist(), n_rois=cfg["n_rois"], prefix=args.prefix,
                         demo=args.demo, roi_matrices_file=None if args.demo else cfg.get("roi_matrices_file"))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        y_true, prob = predict(model, loader, device)
        m = compute_metrics(y_true, prob)
        rows.append(m)
        print(f"fold{fold} (n={len(y_true)}): " + "  ".join(f"{k}={v:.2f}" for k, v in m.items()), flush=True)

    print("=== mean +- std over folds (%) ===")
    print(summarize(rows))
    if args.demo:
        print("(synthetic demo samples: the numbers are not meaningful)")


if __name__ == "__main__":
    main()
