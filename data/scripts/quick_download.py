#!/usr/bin/env python3
"""
Quick dataset downloader for immediate results
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.datasets import load_breast_cancer
import requests
from tqdm import tqdm

def main():
    base_path = Path(".")
    raw_path = base_path / "raw"
    processed_path = base_path / "processed"
    
    raw_path.mkdir(exist_ok=True)
    processed_path.mkdir(exist_ok=True)
    
    datasets = []
    
    print("=== Quick Cancer Dataset Download ===")
    
    # 1. Wisconsin Breast Cancer Dataset
    print("1. Downloading Wisconsin Breast Cancer dataset...")
    wisconsin_dir = processed_path / "wisconsin_breast_cancer"
    wisconsin_dir.mkdir(exist_ok=True)
    
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['target_names'] = [data.target_names[i] for i in data.target]
    df.to_csv(wisconsin_dir / "breast_cancer_wisconsin.csv", index=False)
    
    with open(wisconsin_dir / "README.md", 'w') as f:
        f.write(f"# Wisconsin Breast Cancer Dataset\n\n{data.DESCR}")
    
    datasets.append({
        'name': 'Wisconsin Breast Cancer',
        'source': 'Scikit-learn',
        'path': str(wisconsin_dir),
        'samples': len(df),
        'features': len(data.feature_names),
        'description': '569 breast cancer cases with 30 diagnostic features'
    })
    
    # 2. Synthetic Cancer Images
    print("2. Generating synthetic cancer images...")
    synthetic_dir = processed_path / "synthetic_cancer"
    synthetic_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    labels = []
    
    for category in ['benign', 'malignant']:
        cat_dir = synthetic_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        for i in range(50):  # 50 per category for speed
            # Generate synthetic medical image
            img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            
            # Add circular pattern simulating lesion
            center_x, center_y = 32, 32
            radius = np.random.randint(10, 25)
            
            y, x = np.ogrid[:64, :64]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            if category == 'malignant':
                img_data[mask] = [255, 100, 100]  # Reddish
            else:
                img_data[mask] = [100, 255, 100]  # Greenish
            
            img = Image.fromarray(img_data)
            img.save(cat_dir / f"{category}_{i:03d}.jpg")
            
            labels.append({
                'filename': f"{category}/{category}_{i:03d}.jpg",
                'label': category,
                'category': 1 if category == 'malignant' else 0
            })
    
    pd.DataFrame(labels).to_csv(synthetic_dir / "labels.csv", index=False)
    
    datasets.append({
        'name': 'Synthetic Cancer Images',
        'source': 'Generated',
        'path': str(synthetic_dir),
        'samples': 100,
        'features': 'RGB Images (64x64)',
        'description': '100 synthetic medical images for INR testing'
    })
    
    # 3. Download small skin cancer metadata
    print("3. Downloading HAM10000 metadata...")
    ham_dir = raw_path / "ham10000_metadata"
    ham_dir.mkdir(exist_ok=True)
    
    try:
        metadata_url = "https://dataverse.harvard.edu/api/access/datafile/3450625"
        response = requests.get(metadata_url)
        if response.status_code == 200:
            with open(ham_dir / "HAM10000_metadata.csv", 'wb') as f:
                f.write(response.content)
            
            datasets.append({
                'name': 'HAM10000 Metadata',
                'source': 'Harvard Dataverse',
                'path': str(ham_dir),
                'samples': '10,015',
                'features': 'Metadata fields',
                'description': 'Skin lesion metadata for dermoscopy analysis'
            })
        else:
            print(f"Failed to download HAM10000 metadata: {response.status_code}")
    except Exception as e:
        print(f"Error downloading HAM10000: {e}")
    
    # 4. Create sample brain tumor data
    print("4. Creating sample brain tumor dataset...")
    brain_dir = processed_path / "sample_brain_tumor"
    brain_dir.mkdir(exist_ok=True)
    
    # Create synthetic brain MRI-like data
    brain_labels = []
    categories = ['no_tumor', 'glioma', 'meningioma', 'pituitary']
    
    for category in categories:
        cat_dir = brain_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        for i in range(25):  # 25 per category
            # Generate brain-like image
            img_data = np.random.randint(20, 80, (128, 128), dtype=np.uint8)
            
            # Add brain-like structure
            center = (64, 64)
            for r in range(10, 50, 5):
                y, x = np.ogrid[:128, :128]
                mask = (x - center[0])**2 + (y - center[1])**2 <= r**2
                img_data[mask] += np.random.randint(10, 30)
            
            # Add tumor-like features for non-normal cases
            if category != 'no_tumor':
                tumor_x = np.random.randint(30, 98)
                tumor_y = np.random.randint(30, 98)
                tumor_radius = np.random.randint(5, 15)
                
                y, x = np.ogrid[:128, :128]
                mask = (x - tumor_x)**2 + (y - tumor_y)**2 <= tumor_radius**2
                img_data[mask] = 255
            
            img_data = np.clip(img_data, 0, 255)
            img = Image.fromarray(img_data, mode='L')
            img.save(cat_dir / f"{category}_{i:03d}.jpg")
            
            brain_labels.append({
                'filename': f"{category}/{category}_{i:03d}.jpg",
                'label': category,
                'tumor_type': category
            })
    
    pd.DataFrame(brain_labels).to_csv(brain_dir / "labels.csv", index=False)
    
    datasets.append({
        'name': 'Sample Brain Tumor',
        'source': 'Generated',
        'path': str(brain_dir),
        'samples': 100,
        'features': 'Grayscale Images (128x128)',
        'description': '100 synthetic brain MRI images across 4 categories'
    })
    
    # 5. Create summary
    print("5. Creating dataset summary...")
    summary_df = pd.DataFrame(datasets)
    summary_df.to_csv(base_path / "dataset_summary.csv", index=False)
    
    # Create detailed README
    total_samples = sum(int(str(d['samples']).replace(',', '').split()[0]) for d in datasets if str(d['samples']).replace(',', '').split()[0].isdigit())
    
    readme_content = f"""# Medical Imaging Cancer Detection Datasets

## Overview
This repository contains {len(datasets)} datasets for cancer detection research using Implicit Neural Representations (INRs).

**Total Samples**: {total_samples:,} across all datasets

## Research Framework
These datasets support testing INR architectures for cancer detection:
- **{len(datasets)} datasets** × **3 architectures** × **5 seeds** = **{len(datasets) * 3 * 5} experiments**

## Dataset Summary

| Dataset | Source | Samples | Features | Description |
|---------|--------|---------|----------|-------------|
"""
    
    for dataset in datasets:
        readme_content += f"| {dataset['name']} | {dataset['source']} | {dataset['samples']} | {dataset['features']} | {dataset['description']} |\n"
    
    readme_content += f"""

## Usage Instructions

### Loading Datasets

```python
import pandas as pd
from PIL import Image
import numpy as np

# Load Wisconsin Breast Cancer data
wisconsin_data = pd.read_csv('processed/wisconsin_breast_cancer/breast_cancer_wisconsin.csv')
print(f"Wisconsin dataset shape: {{wisconsin_data.shape}}")

# Load synthetic cancer images
synthetic_labels = pd.read_csv('processed/synthetic_cancer/labels.csv')
img = Image.open('processed/synthetic_cancer/benign/benign_000.jpg')
print(f"Synthetic image size: {{img.size}}")

# Load brain tumor samples
brain_labels = pd.read_csv('processed/sample_brain_tumor/labels.csv')
brain_img = Image.open('processed/sample_brain_tumor/no_tumor/no_tumor_000.jpg')
print(f"Brain image size: {{brain_img.size}}")
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

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(base_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\n=== Download Complete ===")
    print(f"Downloaded {len(datasets)} datasets")
    print(f"Total samples: {total_samples:,}")
    print("\nDataset Summary:")
    for dataset in datasets:
        print(f"  - {dataset['name']}: {dataset['samples']} samples")
    
    return datasets

if __name__ == "__main__":
    main()