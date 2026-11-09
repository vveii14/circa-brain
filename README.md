# CIRCA — Contrastive Imaging–ROI Cross-view Alignment

This repository contains the archived seed-42 ADHD classification checkpoints
and their inference code from `adhd_NG_3DSCTF_seed42_archive`.

## Current checkpoint evaluation

The evaluation entry point imports the archived implementation from `code/`.
The checkpoint configuration uses a **3DSCTF image encoder** and a
**BrainNet ResidualGNNs ROI encoder with GraphSAGE (SAGEConv)**. The archive
name contains `NG`, but the implementation uses the `brainnet` registry key.
Both branches produce 256-dimensional embeddings. Classification uses their
concatenation; the model also defines a symmetric InfoNCE alignment loss.
The archive does not contain the original training loop, so the saved
configuration alone does not establish how that loss was used during training.

```bash
pip install -r requirements.txt
python eval_checkpoints.py
# Optional checkpoint and data directories:
python eval_checkpoints.py checkpoints /path/to/adhd
```

Default data location: `./data/adhd`. Required files:

- `adhd_fold1_775.npy` through `adhd_fold5_775.npy`: NumPy object arrays of
  subject dictionaries containing `raw_img_np` (96 x 96 x 96) and binary `label`.
- `roi_matrices_775.npy`: aligned per-subject 116 x 116 correlation matrices.
  Matrix row order must match concatenation of the five image files in order.

Supply the original prepared data in its original order. The full dataset is
not included. The archived loader can fall back to ROI vectors when the matrix
file is absent; that changes the model input and is not a valid substitute for
reproducing the matrix-based checkpoint evaluation.

The evaluator concatenates the five files and reconstructs
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. It applies a fixed
classification threshold of 0.5 without fitting a threshold during evaluation.
It prints Accuracy, AUC, Sensitivity, Specificity and macro-F1 in percent,
including the five-fold mean and sample standard deviation; argmax results
are also printed. These are reconstructed validation-fold results, not evidence
of an independent test set. Original split IDs and training logs are not in the
archive, so exact split correspondence and checkpoint-selection independence
cannot be verified from this archive alone.

### Saved checkpoint metadata

These values are stored in the checkpoints, not freshly measured results:

| Fold | Saved epoch | Saved validation AUC (%) |
|---|---:|---:|
| 1 | 36 | 83.90 |
| 2 | 35 | 70.96 |
| 3 | 33 | 71.12 |
| 4 | 39 | 70.00 |
| 5 | 27 | 70.84 |

Mean saved validation AUC: 73.37 +/- 5.90 percentage points (sample SD).

## Archive integrity and layout

- `checkpoints/`: the five archived weights.
- `code/`: the archived model, dataset and inference utility implementations.
- `eval_checkpoints.py`: archive evaluator with portable repository-local defaults
  and the existing optional `CHECKPOINT_DIR DATA_DIR` command-line interface.
- `MD5SUMS`: checksums for the current inference release; run `md5sum -c MD5SUMS`.
- `ARCHIVE_MD5SUMS`: original archive manifest for provenance. Its evaluator hash
  differs because of the documented path/interface adaptation above.

The checkpoint files are byte-identical to the previous repository version;
this update restores the archive inference code and corrects the documentation.

## Earlier training and demo code

The root-level `train_kfold.py`, `models/`, `data/`, and `config.yaml` are retained
from the earlier release. They are separate from the archived inference code
under `code/`; the current evaluator does not import the earlier training script.
The bundled ten-subject demo is retained for that earlier pipeline, and cannot
reproduce the full-data checkpoint metrics. The earlier training pipeline has
not been verified as the original producer of these archived weights.
