# Cluster-Based-Transfer-Learning-for-Brain-Computer-Interfaces
**Reducing Calibration Effort via Population Structure and Few-Shot Adaptation**

---

## Overview

This repository contains the complete experimental pipeline developed for my MSc thesis:

**“Cluster-Based Transfer Learning for Motor Imagery BCI:  
Evaluating Offline Generalization and In-Session Calibration”**

The project addresses a central bottleneck in EEG-based Brain–Computer Interfaces (BCIs):  
**poor cross-subject generalization and the need for extensive per-user calibration**.

The core idea is to explicitly model **population structure** and use it to guide transfer learning.  
By combining **subject clustering**, **multi-task learning**, and **lightweight few-shot adaptation**,  
the framework enables **accurate motor imagery decoding with only a handful of calibration trials**.

## Key Contributions

### Cluster-Aware Transfer Learning Pipeline
- Learns shared representations across a large population
- Stratifies subjects into homogeneous clusters using EEG-derived features
- Conditions model heads on cluster identity to reduce negative transfer

### Multi-Task & Transfer Learning Integration
- Shared Deep4Net backbone trained across subjects
- Cluster-specific lightweight heads
- Supports:
  - Pooled transfer
  - LOSO zero-shot transfer
  - LOSO few-shot calibration (4 trials per class)

### Deployment-Oriented Evaluation
- Trained on a harmonized 85-subject motor imagery dataset
- Tested on prospectively recorded EEG using a consumer-grade headset
- Strict separation between training, calibration, and testing
- No information leakage across subjects

### Data-Efficient Personalization
- Few-shot calibration consistently outperforms pooled baselines
- Zero-shot transfer alone is insufficient
- Demonstrates that targeted adaptation beats generic augmentation

## What’s Included

### EEG Processing & Feature Pipeline
- Unified preprocessing for heterogeneous datasets
- Sliding-window epoching with leakage-safe standardization
- Feature families:
  - CSP (primary)
  - ERD/ERS
  - FBCSP
  - Riemannian features
- Fixed train-only transforms reused for unseen users

### Population Clustering
- Subject embeddings derived from pooled training data
- k-means clustering (default: k = 3, CSP-based)
- Stability validated via silhouette scores and ARI
- Cluster assignment reused during transfer and personalization

### Models
- Deep4Net baseline (raw EEG CNN)
- Multi-task model with shared backbone and cluster-conditioned heads
- Transfer learning model for zero- and few-shot adaptation

### Evaluation Protocols
- Leave-One-Subject-Out (LOSO)
- Zero-shot vs few-shot comparison
- Calibration ablations (number of trials, cluster restriction)
- Accuracy, Cohen’s κ, precision, recall, and F1-score

## Repository Structure

```text
.
├── config/                 # Hydra configuration (datasets, experiments, models)
├── src/                    # Entry points & experiment runners
├── lib/                    # Core pipeline, models, trainers, evaluation
├── requirements.txt
└── README.md
```
The entire pipeline is Hydra-driven, enabling reproducible experiments via configuration composition.

## Quickstart

### 1) Create the environment

**Option A – Conda**
```bash
conda env create -f environment.yml
conda activate bci-transfer
```

**Option B – pip**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run experiments
`python src/main.py experiment=transfer_few_shot`
All preprocessing statistics, clustering, and model selection are derived strictly from training data.

## Real-World Impact
- Demonstrates how population structure improves transfer learning
- Shows that more calibration data does not necessarily yield better performance
- Provides a deployment-relevant pathway for motor imagery BCIs
- Bridges academic rigor with practical constraints such as low channel count and limited calibration data

This repository forms the technical foundation of my MSc thesis and represents a complete, reusable framework for subject-aware EEG transfer learning.