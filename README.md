# CIRCA: Contrastive Imaging–ROI Cross-view Alignment

Official PyTorch implementation of

> **Joint Imaging–ROI Representation Learning via Cross-View Contrastive Alignment for Brain Disorder Classification**
> Wei Liang, Lifang He. *MICCAI 2026.*

CIRCA learns a subject-level embedding of the whole 3D brain volume (imaging branch) and an embedding of the ROI connectivity graph (ROI branch). A bidirectional InfoNCE objective aligns the two embeddings in a shared space, and the concatenated embeddings are used for classification.

| Component | Module | Input → output |
|---|---|---|
| Imaging branch | 3DSC-TF (3D depthwise-separable CNN + Transformer) | `(B, 1, 96, 96, 96)` → `(B, 256)` |
| ROI branch | NeuroGraph residual GNN (GraphSAGE layers) | `(B, 116, 116)` → `(B, 256)` |
| Fusion | Cross-view contrastive alignment + MLP classifier | `(B, 256) + (B, 256)` → `(B, 2)` |

The ROI branch turns each connectivity matrix into a 116-node graph: node features are the matrix rows, and the edges are the strongest 12.7% of upper-triangular entries (`edge_top_p` in `config.yaml`).

This repository includes the training code, the five ADHD-200 checkpoints (one per fold) and a few synthetic samples for checking the installation.

## Installation

```bash
git clone https://github.com/vveii14/circa-brain.git
cd circa-brain
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.10 (CUDA 12.8) and PyTorch Geometric 2.7. Install the PyTorch and PyTorch Geometric builds that match your CUDA version first if `pip` picks the wrong ones.

## Data

ADHD-200 and ABIDE are public datasets but we cannot redistribute them. Request access from the original providers and convert the data to the format below.

```text
data/adhd/
├── adhd_fold1_<N>.npy  ...  adhd_fold5_<N>.npy   # subject records, 5 files
└── roi_matrices_<N>.npy                          # (N, 116, 116) float32
```

`<N>` is the total number of subjects. The loader reads it from the fold-file names. ABIDE uses the same layout with the prefix `abide` (pass `--prefix abide`).

**Subject records.** Each fold file is a NumPy array (`allow_pickle=True`). Element `i` must provide:

| Field | Shape / type | Content |
|---|---|---|
| `raw_img_np` | `(96, 96, 96)` float32 | T1-weighted volume from the fMRIPrep release: skull-stripped, in MNI space, cropped, resized to 96³ and scaled to [0, 1] |
| `label` | int | 0 = typically developing control, 1 = patient (ADHD or ASD) |

**ROI connectivity.** Row `i` of `roi_matrices_<N>.npy` is the 116 × 116 ROI connectivity matrix of subject `i` over the AAL-116 atlas regions. Subjects are ordered as in the concatenation of `fold1` … `fold5`, so the order of the two files must match.

**Splits.** The five fold files are concatenated and then re-split with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, stratified by diagnosis at the subject level. In fold *k*, part *k* is the test set, part *k*+1 is the validation set and the other three parts are the training set (`utils.make_splits`).

**Synthetic demo samples.** `data/adhd/adhd_demo10.npy` and `data/adhd/roi_matrices_demo10.npy` contain 10 **randomly generated** samples in the same format. They are not real subjects. Use them only to check that the code runs. Metrics computed on them mean nothing.

## Evaluate the released checkpoints

`checkpoints/fold{k}_best.pth` contains only the model weights (a `state_dict`). The architecture is read from `config.yaml`.

```bash
python eval_checkpoints.py --data_dir data/adhd   # full ADHD-200 data, 5 folds
python eval_checkpoints.py --demo                 # installation check on the synthetic samples
```

Each checkpoint is evaluated on the test part of its fold. On our prepared ADHD-200 data, the output is:

| Fold | Acc (%) | AUC (%) | Sens (%) | Spec (%) | F1 (%) |
|---|---|---|---|---|---|
| 1 | 78.71 | 83.90 | 86.15 | 73.33 | 78.62 |
| 2 | 61.29 | 70.96 | 92.19 | 39.56 | 60.42 |
| 3 | 63.23 | 71.12 | 78.12 | 52.75 | 63.22 |
| 4 | 62.58 | 70.00 | 90.62 | 42.86 | 62.01 |
| 5 | 62.58 | 70.84 | 75.00 | 53.85 | 62.58 |
| **Mean ± std** | **65.68 ± 7.32** | **73.37 ± 5.90** | **84.42 ± 7.59** | **52.47 ± 13.19** | **65.37 ± 7.48** |

Predictions take the argmax over the two classes. F1 is macro-averaged.

## Train

```bash
# Proposed model: 3DSC-TF + NeuroGraph with cross-view contrastive alignment
python train_kfold.py --config config.yaml --data_dir data/adhd --output_dir runs/adhd_contrastive

# Ablations with the same backbones
python train_kfold.py --config config.yaml --data_dir data/adhd --fusion concat   --output_dir runs/adhd_concat
python train_kfold.py --config config.yaml --data_dir data/adhd --mode image_only --output_dir runs/adhd_image_only
python train_kfold.py --config config.yaml --data_dir data/adhd --mode roi_only   --output_dir runs/adhd_roi_only

# Installation check on the synthetic samples (a few seconds)
python train_kfold.py --config config.yaml --demo --epochs 2 --batch_size 4 --output_dir runs/demo
```

For each fold, the script keeps the epoch with the best validation AUC, evaluates it on the test set, saves its weights to `<output_dir>/checkpoints/fold{k}_best.pth` and writes the log to `<output_dir>/train.log`. At the end it prints the mean ± std over the five folds. `config.yaml` holds the hyperparameters of the released model. Runs are seeded, and cuDNN/cuBLAS are set to deterministic mode.

Training on the full ADHD-200 data loads about 9 GB of images into memory. For faster loading you can copy `data/adhd` to a RAM disk (for example `/dev/shm`) and pass that path as `--data_dir`.

## Repository structure

```text
circa-brain/
├── config.yaml                  # hyperparameters of the released model
├── train_kfold.py               # 5-fold training and testing
├── eval_checkpoints.py          # evaluation of the released checkpoints
├── utils.py                     # splits, model construction, metrics
├── data/
│   ├── adhd.py                  # dataset and loader (ADHD-200 / ABIDE)
│   └── adhd/                    # synthetic demo samples; put the real data here
├── models/
│   ├── fusion_model.py          # imaging branch + ROI branch + fusion head
│   ├── fusion.py                # contrastive fusion (proposed) and concat fusion
│   ├── image_encoder_3dsctf.py  # 3DSC-TF imaging encoder
│   ├── roi_encoder_neurograph.py
│   ├── neurograph_residual_gnn.py
│   ├── registry.py, _register.py
├── checkpoints/fold{1..5}_best.pth
├── MD5SUMS
└── requirements.txt
```

## Citation

```bibtex
@inproceedings{liang2026circa,
  title     = {Joint Imaging--ROI Representation Learning via Cross-View Contrastive Alignment for Brain Disorder Classification},
  author    = {Liang, Wei and He, Lifang},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

## Acknowledgements

- The imaging encoder is adapted from [3DSC-TF](https://github.com/NWPU-903PR/3DSC-TF) (*Classification of Alzheimer's Disease by Jointing 3D Depthwise Separable Convolutional Neural Network and Transformer*).
- The ROI encoder uses the ResidualGNNs backbone from [NeuroGraph](https://github.com/Anwar-Said/NeuroGraph) (MIT License).
- We thank the ADHD-200 and ABIDE consortia for making their data available.

## License

The code in this repository is released under the [MIT License](LICENSE). Code adapted from third-party projects is subject to the licenses of those projects.
