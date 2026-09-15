# BoneMTL

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Multitask Deep Learning for Classification and Segmentation of Bone Tumors in X-ray Images**

BoneMTL is a multi-task learning framework that jointly performs pixel-level segmentation and three-tier hierarchical classification of bone tumors from X-ray images using the BTXRD dataset.

---

## Results

All metrics reported on the held-out test set using Stochastic Weight Averaging (SWA).

| Task | Metric | Score |
|------|--------|-------|
| Segmentation | Dice | **0.5391** |
| Segmentation | IoU | **0.4258** |
| Tier 1 (Tumor Detection) | F1 | **0.8231** |
| Tier 1 | AUC | **0.9089** |
| Tier 1 | Accuracy | **0.8310** |
| Tier 2 (Malignancy) | F1 | **0.7917** |
| Tier 2 | AUC | **0.9648** |
| Tier 2 | Accuracy | **0.9278** |
| Tier 3 (9 Tumor Types) | F1 macro | **0.4563** |
| Tier 3 | Accuracy | **0.6245** |

---

## Architecture

```text
Input (3, 256, 256)
         ↓
ResNet50 Encoder (pretrained ImageNet, shared)
         ↓                    ↓
Classification Head    U-Net Decoder + Deep Supervision
    GAP → 512              aux3 (dec3), aux2 (dec2)
    fc_tier1 (1)           seg_out (1, 256, 256)
    fc_tier2 (1)
    fc_tier3 (9)

Total parameters: 33,573,566
```

---

## Key Contributions

- **Hierarchical multi-task loss**: BCE (Tier 1), weighted BCE (Tier 2), Focal Loss γ=2.0 (Tier 3), Dice+BCE (segmentation)
- **Deep Supervision**: auxiliary segmentation heads at dec3 and dec2 improve gradient flow
- **Stochastic Weight Averaging (SWA)**: model weight averaging from epoch 60 improves generalization
- **Strong augmentation**: ElasticTransform, GridDistortion, CoarseDropout for small dataset generalization
- **Grad-CAM visualization**: explainability for clinical interpretability

---

## Dataset

BTXRD — Vietnamese bone tumor X-ray dataset (not included, private).

| Split | Images |
|-------|--------|
| Train | 2,622 |
| Val   | 562   |
| Test  | 562   |

Three-tier label hierarchy:
- **Tier 1**: tumor / no tumor (binary)
- **Tier 2**: benign / malignant (binary)
- **Tier 3**: 9 tumor types (multi-class)

---

## Installation

```bash
conda create -n bonemtl python=3.11
conda activate bonemtl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Usage

**Train:**
```bash
python train.py
```

**Evaluate on test set:**
```bash
python evaluate.py
```

**Visualize results + Grad-CAM:**
```bash
python visualize.py
```

**Per-class analysis + confusion matrices:**
```bash
python analyze.py
```

---

## Training Config

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet50 (ImageNet pretrained) |
| Optimizer | Adam, lr=1e-4 |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Batch size | 8 |
| Image size | 256×256 |
| λ_seg | 2.0 |
| Focal γ | 2.0 |
| λ_aux (Deep Supervision) | 0.4 |
| SWA start | epoch 60 |
| Early stopping | patience=10 |

---

## Visualization

![Results](results/visualization.png)

*Columns: X-ray Input | GT Mask | Pred Mask | Seg Overlay | Grad-CAM | Classification*

*Images blurred for privacy.*

---

## Project Structure

```text
BoneMTL/
├── src/
│   ├── dataset.py     # BTXRDDataset
│   ├── model.py       # BoneMTL, Deep Supervision heads
│   ├── losses.py      # FocalLoss, MultiTaskLoss
│   ├── metrics.py     # Dice, IoU, F1, AUC
│   ├── trainer.py     # Training loop, SWA, Mixup
│   └── utils.py       # Checkpointing, config loading
├── configs/
│   └── default.yaml
├── report/
│   └── index.qmd      # Quarto manuscript
├── results/           # Metrics, visualizations
├── train.py
├── evaluate.py
├── visualize.py
├── analyze.py
└── requirements.txt
```

---

## Citation

```bibtex
@misc{dotranminhnhat2026bonemtl,
  title   = {BoneMTL: Multitask Deep Learning for Classification and Segmentation of Bone Tumors},
  author  = {Do Tran Minh Nhat},
  year    = {2026},
  school  = {VNU-HCMUS}
}
```