# FJNNet: Deep Learning for Scattering Imaging

This repository contains the official PyTorch implementation of the paper: **"Scattering Imaging Beyond the Optical Memory Effect Using Deep Learning Model
Trained on Physically Inspired Synthetic Data"**. 

---

## Project Structure

```text
├── network/            # Network architecture
│   ├── FJNNET.py       # Main model definition
│   └── layers.py       # Custom layers and attention modules
├── dataset/            # Data loading
│   └── dataset.py      
├── utils/              # tools
│   └── utils.py        
├── preprocess          # Matlab script for background correction & denoising
│   └── preprocess.m
├── train.py            # train script (Supports Training & Evaluation use DDP)
└── requirements.txt    # Environment dependencies
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Garankeil/FJNNet.git
cd FJNNet
```

### 2. Install dependencies
Ensure you have Python 3.8+ and CUDA installed.
```bash
pip install -r requirements.txt
```

---

## Data Preparation

Please organize your dataset patterns as follows:

```text
./data/
  ├── train/
  │   ├── speckle/   # Input speckle patterns (such as: "sp_1.png")
  │   └── label/     # Ground truth labels (such as: "lb_.png")
  ├── eval/
  │   ├── speckle/
  │   └── label/
```

*Note: The dataset loader sorts files numerically based on the filename indices.*

---

## Usage

This project utilizes **Distributed Data Parallel (DDP)** for accelerated training.

### 1. Training
Run the following command to start training on 2 GPUs:

```bash
torchrun --nproc_per_node=2 train.py \
    --train True \
    --train-root ./data/train \
    --eval-root ./data/eval \
    --save-path ./checkpoints \
    --epochs 100 \
    --batch-size 12
```

### 2. Evaluation Only
To test a pre-trained model and save reconstructed results:

```bash
torchrun --nproc_per_node=1 train.py \
    --train False \
    --model-path ./checkpoints/best_model.pt \
    --eval-root ./data/eval
```
*Results will be saved in `./results/recon/`.*

---

## Preprocessing (Matlab)
We provide `preprocess.m` for raw data preprocessing:
- **Background Correction**: Normalizes non-uniform illumination.
- **Denoising**: Applies $384 \to 64 \to 384$ interpolation-based smoothing.

---
