#!/usr/bin/env python3
"""
Dataset Download Script for Cancer Detection Research
Downloads medical imaging datasets for cancer detection research
"""

import os
import requests
import tarfile
import zipfile
import gdown
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_blobs
import torch
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import json

def create_synthetic_medical_dataset():
    """Create synthetic medical imaging-like dataset for demonstration"""
    print("Creating synthetic medical imaging dataset...")
    
    # Create synthetic tabular medical data (patient features)
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create feature names representing medical measurements
    feature_names = [
        'age', 'bmi', 'tumor_size_mm', 'lymph_nodes_positive',
        'estrogen_receptor', 'progesterone_receptor', 'her2_status',
        'grade', 'stage', 'ki67_percent', 'p53_mutation',
        'brca1_mutation', 'brca2_mutation', 'menopause_status',
        'family_history', 'smoking_history', 'alcohol_consumption',
        'physical_activity', 'contraceptive_use', 'hormone_therapy'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['cancer_diagnosis'] = y  # 0: benign, 1: malignant
    
    # Add some realistic medical value ranges
    df['age'] = np.clip(df['age'] * 20 + 50, 25, 85).astype(int)
    df['bmi'] = np.clip(df['bmi'] * 5 + 25, 18, 45)
    df['tumor_size_mm'] = np.clip(np.abs(df['tumor_size_mm']) * 10 + 5, 2, 50)
    df['lymph_nodes_positive'] = np.clip(np.abs(df['lymph_nodes_positive']) * 2, 0, 10).astype(int)
    
    # Save synthetic dataset
    synthetic_path = Path('data/processed/synthetic_cancer_data.csv')
    df.to_csv(synthetic_path, index=False)
    
    print(f"✓ Synthetic medical dataset created: {synthetic_path}")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Classes: Benign ({(y==0).sum()}), Malignant ({(y==1).sum()})")
    
    return df

def download_medical_sample_data():
    """Download sample medical datasets from public sources"""
    print("Downloading sample medical datasets...")
    
    # Download Wisconsin Breast Cancer Dataset (small, good for testing)
    from sklearn.datasets import load_breast_cancer
    
    cancer_data = load_breast_cancer()
    
    # Create DataFrame
    df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    df['target'] = cancer_data.target
    df['target_names'] = [cancer_data.target_names[i] for i in cancer_data.target]
    
    # Save dataset
    wisconsin_path = Path('data/processed/wisconsin_breast_cancer.csv')
    df.to_csv(wisconsin_path, index=False)
    
    print(f"✓ Wisconsin Breast Cancer dataset downloaded: {wisconsin_path}")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(cancer_data.feature_names)}")
    print(f"  - Classes: {dict(zip(cancer_data.target_names, np.bincount(cancer_data.target)))}")
    
    return df

def create_image_samples():
    """Create sample medical-like images"""
    print("Creating sample medical imaging data...")
    
    # Create synthetic 'medical' images (grayscale patterns)
    img_dir = Path('data/processed/synthetic_medical_images')
    img_dir.mkdir(exist_ok=True)
    
    # Create normal vs abnormal image patterns
    for category in ['normal', 'abnormal']:
        cat_dir = img_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        for i in range(50):  # 50 images per category
            # Create synthetic medical-like patterns
            if category == 'normal':
                # Regular, symmetric pattern
                img = np.random.normal(0.3, 0.1, (128, 128))
                img += 0.2 * np.sin(np.linspace(0, 4*np.pi, 128))[:, np.newaxis]
            else:
                # Irregular, asymmetric pattern (simulating abnormalities)
                img = np.random.normal(0.5, 0.2, (128, 128))
                # Add "tumor-like" circular regions
                center_x, center_y = np.random.randint(30, 98, 2)
                radius = np.random.randint(10, 25)
                y, x = np.ogrid[:128, :128]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                img[mask] += np.random.normal(0.3, 0.1)
            
            # Normalize and save
            img = np.clip((img - img.min()) / (img.max() - img.min()), 0, 1)
            plt.imsave(cat_dir / f'{category}_{i:03d}.png', img, cmap='gray')
    
    print(f"✓ Synthetic medical images created: {img_dir}")
    print(f"  - Normal images: 50")
    print(f"  - Abnormal images: 50")
    print(f"  - Image size: 128x128 pixels")
    
    return img_dir

def download_public_datasets():
    """Download additional public datasets"""
    print("Attempting to download additional public datasets...")
    
    datasets_info = []
    
    # Try to download some public medical datasets
    try:
        # Download MNIST as a proxy for medical imaging (same preprocessing pipeline)
        print("Downloading MNIST dataset (medical imaging preprocessing pipeline test)...")
        mnist_path = Path('data/raw/mnist')
        mnist_path.mkdir(exist_ok=True)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download MNIST
        train_dataset = datasets.MNIST(
            root=str(mnist_path), 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=str(mnist_path), 
            train=False, 
            download=True, 
            transform=transform
        )
        
        print(f"✓ MNIST downloaded: {mnist_path}")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")
        
        datasets_info.append({
            'name': 'MNIST',
            'type': 'Image Classification',
            'samples': len(train_dataset) + len(test_dataset),
            'path': str(mnist_path),
            'use_case': 'Medical imaging preprocessing pipeline validation'
        })
        
    except Exception as e:
        print(f"Error downloading MNIST: {e}")
    
    return datasets_info

def create_dataset_metadata():
    """Create comprehensive dataset metadata"""
    print("Creating dataset metadata...")
    
    metadata = {
        "project": "LLM-Powered Cancer Detection from Medical Imaging",
        "datasets": [
            {
                "name": "Synthetic Cancer Data",
                "type": "Tabular",
                "file": "data/processed/synthetic_cancer_data.csv",
                "description": "Synthetic patient data with medical features",
                "samples": 1000,
                "features": 20,
                "classes": ["benign", "malignant"],
                "format": "CSV",
                "use_case": "Algorithm development and testing"
            },
            {
                "name": "Wisconsin Breast Cancer",
                "type": "Tabular", 
                "file": "data/processed/wisconsin_breast_cancer.csv",
                "description": "Real breast cancer diagnosis dataset from Wisconsin",
                "samples": 569,
                "features": 30,
                "classes": ["malignant", "benign"],
                "format": "CSV",
                "use_case": "Benchmarking and validation"
            },
            {
                "name": "Synthetic Medical Images",
                "type": "Images",
                "file": "data/processed/synthetic_medical_images/",
                "description": "Synthetic medical-like grayscale images",
                "samples": 100,
                "classes": ["normal", "abnormal"],
                "format": "PNG",
                "image_size": "128x128",
                "use_case": "Image processing pipeline development"
            }
        ],
        "download_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_size_mb": 0,
        "notes": [
            "All datasets are for research use only",
            "Synthetic data created for development purposes",
            "Wisconsin dataset is public domain",
            "Real medical datasets require appropriate ethical approval"
        ]
    }
    
    # Save metadata
    metadata_path = Path('data/datasets_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Dataset metadata created: {metadata_path}")
    
    return metadata

def main():
    """Main download function"""
    print("=== Cancer Detection Dataset Download Script ===")
    print("Downloading and organizing datasets for research...\n")
    
    # Ensure data directories exist
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    all_datasets = []
    
    # Download all datasets
    try:
        # Create synthetic data
        synthetic_df = create_synthetic_medical_dataset()
        all_datasets.append("Synthetic Cancer Data")
        
        # Download Wisconsin dataset
        wisconsin_df = download_medical_sample_data()
        all_datasets.append("Wisconsin Breast Cancer")
        
        # Create sample images
        img_dir = create_image_samples()
        all_datasets.append("Synthetic Medical Images")
        
        # Try to download public datasets
        public_datasets = download_public_datasets()
        all_datasets.extend([d['name'] for d in public_datasets])
        
        # Create metadata
        metadata = create_dataset_metadata()
        
        print("\n=== Download Summary ===")
        print(f"✓ Successfully downloaded {len(all_datasets)} datasets:")
        for dataset in all_datasets:
            print(f"  - {dataset}")
        
        # Calculate total size
        total_size = 0
        for root, dirs, files in os.walk('data'):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        
        total_size_mb = total_size / (1024 * 1024)
        print(f"\n✓ Total data size: {total_size_mb:.2f} MB")
        
        print("\n✓ All datasets downloaded successfully!")
        print("Next steps:")
        print("  1. Review data/datasets_metadata.json for dataset details")
        print("  2. Check data/README.md for usage instructions")
        print("  3. Start with synthetic datasets for initial development")
        
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        raise

if __name__ == "__main__":
    main()