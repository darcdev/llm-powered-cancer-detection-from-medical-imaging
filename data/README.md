# Medical Imaging Cancer Detection Datasets

## Overview
This repository contains 4 datasets for cancer detection research using Implicit Neural Representations (INRs).

**Total Samples**: 10,784 across all datasets

## Research Framework
These datasets support testing INR architectures for cancer detection:
- **4 datasets** × **3 architectures** × **5 seeds** = **60 experiments**

## Dataset Summary

| Dataset | Source | Samples | Features | Description |
|---------|--------|---------|----------|-------------|
| Wisconsin Breast Cancer | Scikit-learn | 569 | 30 | 569 breast cancer cases with 30 diagnostic features |
| Synthetic Cancer Images | Generated | 100 | RGB Images (64x64) | 100 synthetic medical images for INR testing |
| HAM10000 Metadata | Harvard Dataverse | 10,015 | Metadata fields | Skin lesion metadata for dermoscopy analysis |
| Sample Brain Tumor | Generated | 100 | Grayscale Images (128x128) | 100 synthetic brain MRI images across 4 categories |


## Usage Instructions

### Loading Datasets

```python
import pandas as pd
from PIL import Image
import numpy as np

# Load Wisconsin Breast Cancer data
wisconsin_data = pd.read_csv('processed/wisconsin_breast_cancer/breast_cancer_wisconsin.csv')
print(f"Wisconsin dataset shape: {wisconsin_data.shape}")

# Load synthetic cancer images
synthetic_labels = pd.read_csv('processed/synthetic_cancer/labels.csv')
img = Image.open('processed/synthetic_cancer/benign/benign_000.jpg')
print(f"Synthetic image size: {img.size}")

# Load brain tumor samples
brain_labels = pd.read_csv('processed/sample_brain_tumor/labels.csv')
brain_img = Image.open('processed/sample_brain_tumor/no_tumor/no_tumor_000.jpg')
print(f"Brain image size: {brain_img.size}")
```

### Dataset Characteristics

**Wisconsin Breast Cancer**:
- 30 diagnostic features (mean, SE, worst values)
- Binary classification (malignant/benign)
- Perfect for baseline tabular ML models

**Synthetic Cancer Images**:
- RGB images (64×64) with simulated lesions
- Controlled dataset for INR architecture testing
- Ground truth available for all samples

**Sample Brain Tumor**:
- Grayscale images (128×128) simulating MRI
- 4-class classification problem
- Includes normal and 3 tumor types

**HAM10000 Metadata** (if downloaded):
- Real-world skin lesion metadata
- Links to 10,000+ dermoscopy images
- Multiple diagnostic categories

## Research Applications

### INR Architecture Testing
1. **SIREN**: Test on synthetic cancer images
2. **NeRF-based**: Apply to 3D brain tumor reconstruction  
3. **Hybrid CNN-INR**: Combine with Wisconsin features

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC for binary classification
- Confusion matrices for multi-class
- Computational efficiency metrics

### Expected Outcomes
- Baseline performance on Wisconsin dataset
- INR overfitting analysis on synthetic data
- Scalability testing across image sizes
- Architecture comparison framework

## File Structure
```
data/
├── raw/                    # Original downloaded data
│   └── ham10000_metadata/  # Skin lesion metadata
├── processed/              # Processed datasets
│   ├── wisconsin_breast_cancer/  # Tabular cancer data
│   ├── synthetic_cancer/         # Generated images
│   └── sample_brain_tumor/       # Brain MRI samples
├── dataset_summary.csv     # Dataset overview
└── README.md              # This file
```

## License Information
- Wisconsin Breast Cancer: BSD License (scikit-learn)
- Synthetic Data: Public Domain (generated)
- HAM10000: CC-BY-NC License (Harvard Dataverse)

Generated on: 2025-10-09 15:50:33
