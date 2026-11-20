# CIRCA: Contrastive Imaging–ROI Cross-view Alignment

This repository provides the archived implementation and pretrained checkpoints for **CIRCA (Contrastive Imaging–ROI Cross-view Alignment)**, together with an evaluation script for reproducing the checkpoint-based results.

The release focuses on **checkpoint evaluation** using the original prepared ADHD dataset. It includes five pretrained checkpoints and the corresponding model and inference implementations. The full dataset is not redistributed.

## 1. Model Architecture

CIRCA employs two complementary encoding branches:

- **Imaging branch:** A 3DSCTF encoder processes 3D neuroimaging volumes.
- **ROI branch:** A BrainNet ResidualGNNs encoder with GraphSAGE (`SAGEConv`) processes ROI connectivity matrices.

Both branches produce 256-dimensional embeddings.

The archived checkpoint configuration uses the `brainnet` registry key for the ROI encoder. Although the archive filename contains `NG`, the underlying implementation uses the BrainNet architecture described above.

## 2. Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 3. Data Preparation

By default, the evaluation script expects the prepared ADHD dataset under:

```text
./data/adhd/
```

The following files are required:

```text
data/adhd/
├── adhd_fold1_775.npy
├── adhd_fold2_775.npy
├── adhd_fold3_775.npy
├── adhd_fold4_775.npy
├── adhd_fold5_775.npy
└── roi_matrices_775.npy
```

### 3.1. Neuroimaging Data

Each `adhd_fold*_775.npy` file contains a NumPy object array of subject-level dictionaries. Each dictionary must include:

- `raw_img_np`: A 3D neuroimaging volume with shape `(96, 96, 96)`.
- `label`: A binary classification label.

### 3.2. ROI Connectivity Data

The `roi_matrices_775.npy` file contains subject-aligned ROI correlation matrices, each with shape `(116, 116)`.

**Subject ordering is essential.** The ROI matrices must follow exactly the same subject order as the concatenation of the five neuroimaging files, from `adhd_fold1_775.npy` through `adhd_fold5_775.npy`.

Use the original prepared data without reordering subjects.

**Important:** The archived data loader supports a fallback to ROI vectors when the connectivity matrix file is unavailable. However, this changes the model input representation and must not be used to reproduce the matrix-based checkpoint evaluation.

## 4. Checkpoint Evaluation

Run the evaluation using the default checkpoint and data directories:

```bash
python eval_checkpoints.py
```

Alternatively, specify custom directories:

```bash
python eval_checkpoints.py checkpoints /path/to/adhd
```

The evaluator loads the five archived checkpoints and reconstructs the validation folds using:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The five neuroimaging files are concatenated in their original order before fold reconstruction.

### Evaluation Metrics

The evaluator reports the following metrics as percentages:

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| AUC | Area under the ROC curve |
| Sensitivity | True positive rate |
| Specificity | True negative rate |
| Macro-F1 | Unweighted mean of per-class F1 scores |

Results are reported for each fold, along with the five-fold mean and sample standard deviation.

Classification predictions use a fixed probability threshold of **0.5**, with no threshold optimization during evaluation. Argmax-based results are also reported for comparison.

**Evaluation scope:** The reported metrics represent reconstructed validation-fold performance based on the archived checkpoints. They should not be interpreted as results from an independent held-out test set.

## 5. Repository Structure

```text
CIRCA/
├── checkpoints/          # Five archived model checkpoints
├── code/                 # Archived model, dataset, and inference implementations
├── eval_checkpoints.py   # Checkpoint evaluation entry point
├── requirements.txt      # Python dependencies
├── MD5SUMS               # Checksums for the current inference release
└── ARCHIVE_MD5SUMS       # Original archive checksums for provenance
```

The evaluation entry point imports the archived implementations from `code/` to preserve consistency with the released checkpoints.

## 6. Archive Integrity

To verify the integrity of the current inference release, run:

```bash
md5sum -c MD5SUMS
```

The repository also includes `ARCHIVE_MD5SUMS`, which preserves the original archive manifest for provenance.

The checksum of `eval_checkpoints.py` differs between the two manifests because the evaluation script was adapted to support repository-relative default paths and optional command-line directory arguments. These changes concern the evaluation interface and file paths.

## 7. Reproducibility Notes

This release is intended to support evaluation of the archived checkpoints rather than end-to-end reproduction of the original training procedure.

For consistent checkpoint evaluation:

- Use the original prepared dataset and preserve subject ordering.
- Provide the aligned `(116, 116)` ROI connectivity matrices.
- Use the reconstructed five-fold splits with `random_state=42`.
- Apply the fixed classification threshold of `0.5`.
- Interpret the reported metrics as reconstructed validation-fold results, not independent test-set performance.

The full prepared dataset is not included in this repository.
