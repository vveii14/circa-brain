# CIRCA · Cross-View Contrastive Imaging–ROI Alignment

Official PyTorch implementation of **"Joint Imaging–ROI Representation Learning via Cross-View
Contrastive Alignment for Brain Disorder Classification"** (MICCAI 2026).

Minimal, self-contained release of the joint **imaging (3D-SCTF) + ROI-graph (NeuroGraph)**
model with cross-view contrastive alignment, for ADHD-200 classification (the
`NeuroGraph + 3DSC-TF` row of the paper's Table 1). 5-fold stratified CV, seed 42.

This folder ships **only** that one model (no other backbones), the **trained checkpoints**,
and a **10-sample demo subset** so the pipeline can be run end-to-end without the full dataset.
Reported metrics: **Accuracy, AUC, F1**.

## Layout

```
config.yaml                  # exact training recipe (hyperparameters)
train_kfold.py               # deterministic 5-fold training (fixed seed -> bit-reproducible)
eval_checkpoints.py          # load the bundled checkpoints, evaluate on the seed-42 val folds
checkpoints/foldN_best.pth   # trained weights (each embeds its own full cfg)
models/                      # 3D-SCTF encoder + NeuroGraph ROI encoder + contrastive fusion
data/adhd/                   # bundled 10-sample DEMO set (+ where the full data goes)
```

## Setup

```bash
pip install -r requirements.txt   # torch, torch-geometric, numpy, scikit-learn, PyYAML
```

## How the data is read

Each subject set is a single NumPy **object array** (`.npy`, load with `allow_pickle=True`);
element `i` is a dict:

```python
{
  "raw_img_np": np.float32 array of shape (96, 96, 96),   # the sMRI volume (image branch input)
  "label":      int,                                       # 0 = control (TDC), 1 = ADHD
  # "raw_img_aal_mask_np": (96,96,96)  # AAL parcellation; only needed if ROI is computed on the fly
}
```

The **ROI branch** input is a precomputed connectivity tensor stored separately:

```
roi_matrices_775.npy   ->  np.float32 array of shape (N, 116, 116)   # 116x116 Pearson matrix per subject
```

Row `i` of the matrix file corresponds to subject `i` of the image array (same order).

**Full dataset layout** (`data/adhd/`):

```
adhd_fold1_775.npy ... adhd_fold5_775.npy   # image dicts; concatenated in order -> 775 subjects
roi_matrices_775.npy                        # (775, 116, 116) connectivity, aligned with the above
```

The loader (`data/adhd.py`) concatenates `adhd_fold1..5` into the full subject set, then
`StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` defines the cross-validation folds.
The image branch reads `raw_img_np`; the ROI branch reads the matching row of `roi_matrices_775.npy`.

**Demo subset** (bundled here): `data/adhd/adhd_test_775.npy` (10 balanced subjects) and
`data/adhd/test_roi_matrices_775.npy` (their 10x116x116 matrices). Selected via `--use_test_set`.

## Run the demo (bundled 10 samples — no extra data needed)

```bash
python train_kfold.py \
    --dataset adhd --mode fusion --fusion contrastive \
    --config config.yaml --use_test_set \
    --n_folds 5 --batch_size 4 --epochs 2 --seed 42 \
    --output_dir logs_demo
```

This trains and evaluates end-to-end on the 10-sample set. It only verifies that the pipeline
runs — with so few samples the scores are not meaningful.

## Train on the full data

Place `adhd_fold1_775.npy ... adhd_fold5_775.npy` and `roi_matrices_775.npy` in `data/adhd/`
(or point `adhd_data_dir` in `config.yaml` elsewhere), then:

```bash
CUDA_VISIBLE_DEVICES=0 python train_kfold.py \
    --dataset adhd --mode fusion --fusion contrastive \
    --config config.yaml --n_folds 5 --seed 42 \
    --output_dir logs_run
```

`--mode image_only` / `--mode roi_only` run the single-branch ablations with the same backbones.

> The full image data is ~9 GB. For fast, stall-free loading, copy it to a RAM disk and set
> `adhd_data_dir: /dev/shm/adhd_data` in `config.yaml`.

## Evaluate the trained checkpoints

`checkpoints/foldN_best.pth` are the trained weights (each embeds its full `cfg`). Evaluating
them reconstructs the seed-42 val folds, so this needs the **full** `data/adhd/` in place:

```bash
python eval_checkpoints.py                       # ./checkpoints + ./data/adhd
python eval_checkpoints.py checkpoints /dev/shm/adhd_data   # RAM-staged data
```

## Notes

- **Reproducibility:** `set_seed()` fixes all RNGs and enables deterministic cuDNN/cuBLAS, so a
  fixed `--seed` yields bit-identical per-fold metrics across runs.
- **Config provenance:** `config.yaml` was recovered verbatim from the embedded `cfg` of
  `checkpoints/fold1_best.pth`. Long training jobs should run inside `tmux`/`nohup`.
