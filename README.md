# BoneMTL

Multitask deep learning for simultaneous hierarchical classification and segmentation of bone tumors in X-ray images.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Architecture

```
Input (3 × 256 × 256)
        ↓
ResNet50 Encoder (shared, ImageNet pretrained)
        ↓
   ┌────┴────┐
   ↓         ↓
Classification   Segmentation
   Head           Decoder (U-Net)
   ↓               ↓
Tier 1 (1,)    Mask (1, 256, 256)
Tier 2 (1,)
Tier 3 (9,)
```

- **Shared encoder**: ResNet50 pretrained on ImageNet, 33.5M parameters
- **Classification head**: 3-tier hierarchical — tumor detection, malignancy assessment, 9-class tumor type identification
- **Segmentation decoder**: U-Net style with skip connections, trained only on samples with pixel-level masks

## Dataset

The dataset used in this project is a private collection of X-ray images related to bone tumors, collected from a medical facility in Vietnam. Due to regulations regarding medical data confidentiality and patient privacy, this dataset cannot be publicly released or distributed.

| Dataset | Images | Masks | Split | Usage |
|---------|--------|-------|-------|-------|
| BTXRD   | 3,746  | 1,867 | 70/15/15 | Train / Val / Test |

If you are interested in accessing the dataset, please contact the author directly. Access is subject to approval.

## Results

| Task | Metric | Score |
|------|--------|-------|
| Segmentation | Dice | 0.4562 |
| Segmentation | IoU | 0.3492 |
| Tier 1 — Tumor detection | F1 | 0.7815 |
| Tier 1 — Tumor detection | AUC | 0.8725 |
| Tier 1 — Tumor detection | Accuracy | 0.7900 |
| Tier 2 — Malignancy | F1 | 0.7414 |
| Tier 2 — Malignancy | AUC | 0.9422 |
| Tier 2 — Malignancy | Accuracy | 0.8936 |
| Tier 3 — Tumor type (9 class) | F1 macro | 0.4334 |
| Tier 3 — Tumor type (9 class) | Accuracy | 0.6135 |

Tier 3 F1 is limited by severe class imbalance (17:1 ratio between most and least frequent tumor types). Segmentation Dice reflects the inherent ambiguity of tumor boundaries in plain X-ray images.

## Visualization

10 sample predictions from the test set.
Each row: input X-ray | ground truth mask | predicted mask | overlay | classification output.
Green text = Tier 1 correct, red text = Tier 1 incorrect.

![Results](results/visualization_blurred.png)

> X-ray images are blurred to protect patient privacy.
> Run `python visualize.py` to generate full quality visualization locally after setting up the data.

## Setup

**Requirements**: Python 3.11, CUDA 11.8, GPU with at least 4GB VRAM.

```bash
git clone https://github.com/DoTranMinhNhat/BoneMTL
cd BoneMTL
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Place the dataset in `data/` with the following structure:

```
data/
├── images/          # X-ray images (.jpeg)
├── masks/           # Segmentation masks (.png)
├── annotations/     # Bounding box annotations (.json)
└── dataset.xlsx     # Labels and metadata
```

## Usage

```bash
# Prepare data splits
python prepare_data.py

# Train
python train.py

# Evaluate best checkpoint
python evaluate.py

# Generate visualization
python visualize.py
```

## Project Structure

```
BoneMTL/
├── src/
│   ├── dataset.py        # BTXRDDataset
│   ├── model.py          # BoneMTL architecture
│   ├── losses.py         # MultiTaskLoss, DiceLoss
│   ├── metrics.py        # Dice, IoU, F1, AUC
│   ├── trainer.py        # Training loop
│   └── utils.py          # Config, seed, checkpoint
├── configs/
│   └── default.yaml      # Hyperparameters
├── tests/
│   ├── test_dataset.py   # 8/8 passed
│   └── test_model.py     # 6/6 passed
├── prepare_data.py
├── train.py
├── evaluate.py
├── visualize.py
└── requirements.txt
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet50 (ImageNet pretrained) |
| Input size | 256 × 256 |
| Batch size | 8 |
| Optimizer | Adam (lr=1e-4, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping | patience=10 |
| GPU | NVIDIA GTX 1650 Max-Q (4GB VRAM) |

## Citation

```bibtex
@article{bonemtl,
  title  = {Multitask Deep Learning for Classification and Segmentation of Bone Tumors in X-ray Images},
  author = {Do Tran Minh Nhat},
}
```