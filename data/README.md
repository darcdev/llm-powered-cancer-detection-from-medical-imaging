# Cancer Detection Medical Imaging Datasets

This folder contains datasets for cancer detection research using medical imaging and machine learning approaches.

## Dataset Overview

### Downloaded Datasets

| Dataset | Type | Samples | Features/Size | Format | Path |
|---------|------|---------|---------------|---------|------|
| **Synthetic Cancer Data** | Tabular | 1,000 | 20 features | CSV | `processed/synthetic_cancer_data.csv` |
| **Wisconsin Breast Cancer** | Tabular | 569 | 30 features | CSV | `processed/wisconsin_breast_cancer.csv` |
| **Synthetic Medical Images** | Images | 100 | 128x128px | PNG | `processed/synthetic_medical_images/` |
| **MNIST** | Images | 70,000 | 28x28px | PyTorch | `raw/mnist/` |

**Total Size:** ~68MB

## Dataset Descriptions

### 1. Synthetic Cancer Data (`synthetic_cancer_data.csv`)
**Purpose:** Algorithm development and testing
**Features:** 20 medical features including:
- Patient demographics (age, BMI)
- Tumor characteristics (size, grade, stage)
- Biomarkers (ER, PR, HER2, Ki67, p53)
- Genetic factors (BRCA1, BRCA2 mutations)
- Risk factors (family history, lifestyle)

**Target:** Binary classification (0=benign, 1=malignant)
**Classes:** Benign: 502, Malignant: 498

### 2. Wisconsin Breast Cancer (`wisconsin_breast_cancer.csv`)
**Purpose:** Benchmarking and validation  
**Features:** 30 real-world diagnostic features computed from digitized breast mass images:
- Radius, texture, perimeter, area, smoothness
- Compactness, concavity, symmetry, fractal dimension
- Statistics: mean, standard error, worst values

**Target:** Binary classification (malignant/benign)
**Classes:** Malignant: 212, Benign: 357
**Source:** UCI Machine Learning Repository

### 3. Synthetic Medical Images (`synthetic_medical_images/`)
**Purpose:** Image processing pipeline development
**Structure:**
```
synthetic_medical_images/
├── normal/           # 50 normal images
└── abnormal/         # 50 abnormal images (with simulated lesions)
```
**Image Properties:**
- Size: 128x128 pixels
- Format: Grayscale PNG
- Normal: Regular, symmetric patterns
- Abnormal: Irregular patterns with circular "tumor-like" regions

### 4. MNIST (`raw/mnist/`)
**Purpose:** Medical imaging preprocessing pipeline validation
**Use Case:** Testing image processing workflows before applying to medical data
- Training: 60,000 images
- Testing: 10,000 images
- Size: 28x28 grayscale

## Loading Data Examples

### Loading Tabular Data
```python
import pandas as pd

# Load synthetic cancer data
synthetic_data = pd.read_csv('data/processed/synthetic_cancer_data.csv')
X_synthetic = synthetic_data.drop('cancer_diagnosis', axis=1)
y_synthetic = synthetic_data['cancer_diagnosis']

# Load Wisconsin breast cancer data
wisconsin_data = pd.read_csv('data/processed/wisconsin_breast_cancer.csv')
X_wisconsin = wisconsin_data.drop(['target', 'target_names'], axis=1)
y_wisconsin = wisconsin_data['target']
```

### Loading Image Data
```python
import torch
from torchvision import datasets, transforms
from PIL import Image
import os

# Load synthetic medical images
def load_synthetic_images(data_dir='data/processed/synthetic_medical_images'):
    images = []
    labels = []
    
    for label, folder in enumerate(['normal', 'abnormal']):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('L')
                images.append(img)
                labels.append(label)
    
    return images, labels

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST(
    root='data/raw/mnist', 
    train=True, 
    transform=transform
)
```

## Research Applications

### 1. INR (Implicit Neural Representation) Architectures
- **Primary Dataset:** Synthetic Medical Images
- **Use Case:** Testing continuous image representation learning
- **Evaluation:** Compare reconstruction quality and efficiency

### 2. Multi-modal Learning
- **Datasets:** Synthetic Cancer Data + Synthetic Medical Images
- **Use Case:** Fusion of tabular clinical data with imaging
- **Approach:** Joint embedding learning

### 3. Transfer Learning
- **Pipeline:** MNIST → Synthetic Medical Images → Real Medical Data
- **Purpose:** Progressive domain adaptation
- **Validation:** Wisconsin Breast Cancer Dataset

## Preprocessing Pipelines

### Image Preprocessing
```python
import torchvision.transforms as transforms

medical_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])
```

### Tabular Data Preprocessing  
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_tabular(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X_scaled, y_encoded, scaler, le
    
    return X_scaled, y, scaler, None
```

## Experimental Design

### Cross-Validation Strategy
- **K-Fold CV:** 5-fold stratified cross-validation
- **Train/Val/Test Split:** 70%/15%/15%
- **Stratification:** Maintain class balance across splits

### Performance Metrics
- **Binary Classification:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Multi-class:** Macro/Micro averaged metrics
- **Regression:** MAE, MSE, R²

### Statistical Testing
- **Significance Tests:** Paired t-tests for model comparison
- **Multiple Comparisons:** Bonferroni correction
- **Effect Size:** Cohen's d for practical significance

## Dataset Limitations & Considerations

### Synthetic Data Limitations
- ⚠️ **Not Real Medical Data:** Synthetic datasets are for development only
- ⚠️ **Distribution Assumptions:** May not reflect real-world complexity
- ⚠️ **Bias:** Limited to programmed patterns and relationships

### Wisconsin Dataset Considerations
- ✅ **Real Data:** Actual diagnostic measurements
- ⚠️ **Age:** Dataset from 1990s, imaging technology has evolved
- ⚠️ **Population:** Limited demographic diversity
- ⚠️ **Size:** Relatively small (569 samples)

### Ethical Considerations
- 🔒 **Privacy:** All data is de-identified or synthetic
- 📋 **IRB Approval:** Required for real medical data usage
- 🎯 **Bias Testing:** Validate across diverse populations
- 📊 **Transparency:** Document all preprocessing steps

## Recommended Public Medical Datasets

For production research, consider these validated medical imaging datasets:

### Large-Scale Public Repositories
1. **The Cancer Imaging Archive (TCIA)**
   - URL: https://www.cancerimagingarchive.net/
   - Content: De-identified medical images, DICOM format
   - Size: Thousands of studies across cancer types

2. **Stanford AIMI Datasets** 
   - URL: https://aimi.stanford.edu/shared-datasets
   - Notable: CheXpert Plus (223k chest X-rays)
   - Content: Curated, annotated medical imaging

3. **RSNA Datasets**
   - Screening Mammography: ~20k studies
   - Pneumonia Detection: 30k chest X-rays
   - Intracranial Hemorrhage: 25k head CT scans

### Domain-Specific Collections
4. **MAMA-MIA** (Nature Scientific Data 2025)
   - Content: 1,506 expert-validated 3D breast DCE-MRI
   - Features: Tumor segmentations, treatment outcomes
   - Use: AI model development and validation

5. **NIH Clinical Center**
   - DeepLesion: 32k lesions from 10k patients
   - Content: Multi-organ, multi-class lesions in CT

## Data Access & Licensing

### Current Datasets
- **Wisconsin Breast Cancer:** Public domain
- **MNIST:** Open source
- **Synthetic Data:** Created for this research

### External Datasets
- Most medical datasets require registration
- Academic use typically free
- Commercial use may require licensing
- Always cite dataset sources in publications

## File Structure
```
data/
├── README.md                          # This file
├── datasets_metadata.json            # Dataset metadata
├── download_datasets.py               # Download script
├── raw/                              # Raw, unprocessed data
│   └── mnist/                        # MNIST dataset files
└── processed/                        # Processed, analysis-ready data
    ├── synthetic_cancer_data.csv     # Synthetic patient data
    ├── wisconsin_breast_cancer.csv   # Wisconsin dataset
    └── synthetic_medical_images/     # Synthetic medical images
        ├── normal/                   # Normal cases
        └── abnormal/                 # Abnormal cases
```

## Next Steps

1. **Data Quality Assessment**
   - Run `scripts/data_quality_check.py` (to be created)
   - Validate data integrity and distributions
   - Generate data quality reports

2. **Baseline Model Development**
   - Start with Wisconsin dataset for benchmarking
   - Implement standard ML baselines (RF, SVM, XGBoost)
   - Establish performance baselines

3. **INR Architecture Testing**  
   - Use synthetic medical images for initial testing
   - Implement continuous image representation
   - Compare with traditional CNN approaches

4. **Real Dataset Integration**
   - Apply for access to TCIA datasets
   - Download domain-specific collections
   - Validate on production medical data

---

**Last Updated:** October 9, 2025  
**Contact:** Research Team  
**License:** Research use only