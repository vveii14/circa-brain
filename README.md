# CIRCA — Contrastive Imaging–ROI Cross-view Alignment

## Current checkpoint evaluation

The evaluation entry point imports the archived implementation from `code/`.
The checkpoint configuration uses a **3DSCTF image encoder** and a
**BrainNet ResidualGNNs ROI encoder with GraphSAGE (SAGEConv)**. The archive
name contains `NG`, but the implementation uses the `brainnet` registry key.
Both branches produce 256-dimensional embeddings. 

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
of an independent test set.

## Archive integrity and layout

- `checkpoints/`: the five archived weights.
- `code/`: the archived model, dataset and inference utility implementations.
- `eval_checkpoints.py`: archive evaluator with portable repository-local defaults
  and the existing optional `CHECKPOINT_DIR DATA_DIR` command-line interface.
- `MD5SUMS`: checksums for the current inference release; run `md5sum -c MD5SUMS`.
- `ARCHIVE_MD5SUMS`: original archive manifest for provenance. Its evaluator hash
  differs because of the documented path/interface adaptation above.
