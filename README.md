# CIRCA — Contrastive Imaging–ROI Cross-view Alignment

Official PyTorch implementation of **"Joint Imaging–ROI Representation Learning via Cross-View
Contrastive Alignment for Brain Disorder Classification"** (MICCAI 2026).

This release contains the single joint model used in the paper — a 3D-SCTF imaging encoder and a
NeuroGraph ROI-graph encoder aligned by a cross-view contrastive objective — together with the
trained ADHD-200 checkpoints and a small bundled subset so the pipeline can be run without the full
dataset.

## Setup

```bash
git clone https://github.com/vveii14/circa-brain.git
cd circa-brain
pip install -r requirements.txt   # torch, torch-geometric, numpy, scikit-learn, PyYAML
```

## Data

The full ADHD-200 / ABIDE data is **not redistributed here** (access is governed by the original
providers). You prepare it yourself into the format below; a 10-sample subset is bundled so you can
verify the code first. Both datasets use the **same format** and AAL-116 atlas; they differ only in
the subject count (and the fold-file suffix).

| Dataset | Image array | Connectivity file | Fold files |
|---|---|---|---|
| ADHD-200 | `(96, 96, 96)` | `roi_matrices_<N>.npy` `(N, 116, 116)` | `adhd_fold1..5_<N>.npy` |
| ABIDE | `(96, 96, 96)` | `roi_matrices_<N>.npy` `(N, 116, 116)` | `abide_fold1..5_<N>.npy` |

`<N>` is the subject count, used as the fold-file suffix (configured per dataset in `data/<dataset>/meta.json`).

### Expected format

Subjects are stored as a NumPy **object array** (`.npy`, `allow_pickle=True`); element `i` is a dict:

```python
{
  "raw_img_np": np.float32 (96, 96, 96),            # imaging-branch input, intensity-normalized to ~[0,1]
  "raw_img_aal_mask_np": np.float32 (96, 96, 96),   # AAL-116 parcellation, aligned to raw_img_np
  "label": int,                                      # 0 = control (TDC), 1 = ADHD / ASD
}
```

The ROI branch consumes a separate per-subject connectivity tensor, aligned by row order:

```
roi_matrices_<N>.npy  ->  np.float32 (N, 116, 116)   # row i is subject i's matrix
```

Each `116 x 116` is the subject's **ROI-by-ROI Pearson correlation matrix** over the 116 AAL regions,
with values in `[-1, 1]`.

> The ROI branch reads **only** `roi_matrices_<N>.npy`. The companion `roi_vectors_<N>.npy` files are
> legacy and unused by this model (the ADHD one is all-zeros) — ignore them.

### Preprocessing (raw scans → the format above)

1. **Imaging volume.** Intensity-normalize each subject's brain volume to `~[0, 1]` and resample to
   `96 x 96 x 96`, giving `raw_img_np`.
2. **AAL parcellation.** Register the AAL-116 atlas onto the same grid to get `raw_img_aal_mask_np`
   (voxel values = region id `1..116`).
3. **Correlation matrix.** For each subject, derive a per-region descriptor from `raw_img_np` over
   the 116 AAL regions and compute their pairwise **Pearson correlation** to form the `116 x 116`
   matrix (`[-1, 1]`); stack across subjects into `roi_matrices_<N>.npy`, in the same order as the
   image array.

The loader (`data/adhd.py`) concatenates `adhd_fold1_<N>.npy ... adhd_fold5_<N>.npy` into the full
subject set and builds the split with `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`.

### What the model receives (per batch of B subjects)

```
Imaging branch (3D-SCTF):  (B, 1, 96, 96, 96)  ->  (B, 256)
ROI branch (NeuroGraph):   (B, 116, 116)        ->  (B, 256)
    each matrix -> a 116-node graph: node features = rows of the matrix (116-d),
    edges = strongest upper-triangular entries (top edge_top_p ~ 0.13)
Fusion (contrastive):  concat (B, 512) -> MLP -> (B, 2) logits,
    plus a contrastive loss aligning the two 256-d embeddings
```

### Bundled demo subset

`data/adhd/adhd_demo10.npy` (10 balanced subjects) and `data/adhd/roi_matrices_demo10.npy`
(`10 x 116 x 116`) are included so you can run the pipeline immediately. Select them with
`--use_test_set`. Ten samples only verify that the code runs; the numbers are not meaningful.

```bash
python train_kfold.py \
    --dataset adhd --mode fusion --fusion contrastive \
    --config config.yaml --use_test_set \
    --n_folds 5 --batch_size 4 --epochs 2 --seed 42 \
    --output_dir logs_demo
```

## Reproduce the paper

Place your prepared `adhd_fold1_<N>.npy ... adhd_fold5_<N>.npy` and `roi_matrices_<N>.npy` in
`data/adhd/` (or point `adhd_data_dir` in `config.yaml` elsewhere), then:

```bash
CUDA_VISIBLE_DEVICES=0 python train_kfold.py \
    --dataset adhd --mode fusion --fusion contrastive \
    --config config.yaml --n_folds 5 --seed 42 \
    --output_dir logs_run
```

`--mode image_only` / `--mode roi_only` run the single-branch ablations with the same backbones.
`config.yaml` is the exact training recipe; `set_seed()` fixes all RNGs and enables deterministic
cuDNN/cuBLAS, so a fixed `--seed` gives the same per-fold results across runs. ABIDE runs identically
with the ABIDE fold files in place.

### Evaluate the trained checkpoints

`checkpoints/foldN_best.pth` are the trained weights (each embeds its full `cfg`). Evaluating them
reconstructs the seed-42 validation folds, so the **full** `data/adhd/` must be in place:

```bash
python eval_checkpoints.py                            # ./checkpoints + ./data/adhd
python eval_checkpoints.py checkpoints /dev/shm/adhd_data   # data staged on a RAM disk
```
