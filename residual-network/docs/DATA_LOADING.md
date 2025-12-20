# EuroSAT Data Loading Guide

## Dataset Structure

The EuroSAT dataset from Kaggle (apollo2506/eurosat-dataset) has the following structure:

```
EuroSAT/
├── AnnualCrop/          (3000 images)
├── Forest/              (3000 images)
├── HerbaceousVegetation/ (3000 images)
├── Highway/             (2500 images)
├── Industrial/          (2500 images)
├── Pasture/             (2000 images)
├── PermanentCrop/       (2500 images)
├── Residential/         (3000 images)
├── River/               (2500 images)
└── SeaLake/             (3000 images)
```

**Total Images**: ~27,000 images across 10 classes

## Data Loading Implementation

### 1. Dataset Download
- Uses `kagglehub.dataset_download("apollo2506/eurosat-dataset")`
- Automatically finds the `EuroSAT` directory (handles nested structures)
- Validates all 10 class directories exist

### 2. Data Splits
- **Train**: 70% (default)
- **Validation**: 15%
- **Test**: 15%
- Splits are reproducible (seed=42)

### 3. Data Augmentation

**Training Transform**:
- Resize to configured size (64x64 default)
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation ±20%)
- Normalize with ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

**Validation/Test Transform**:
- Resize to configured size
- Normalize with ImageNet statistics
- No augmentation

### 4. DataLoader Configuration
- Batch size: 8 (configurable)
- Shuffle: True for training, False for val/test
- Num workers: 4 (configurable)
- Pin memory: False (set True for GPU)

## Class Names (Alphabetically Sorted)

The `torchvision.datasets.ImageFolder` automatically sorts classes alphabetically:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

## Usage

```python
from config import Config
from data_loader import create_data_loaders

config = Config()
train_loader, val_loader, test_loader, class_names = create_data_loaders(config)

# Iterate through data
for images, labels in train_loader:
    # images: (batch_size, 3, img_size, img_size)
    # labels: (batch_size,) with values 0-9
    pass
```

## Key Features

✓ Automatic dataset download from Kaggle  
✓ Handles nested directory structures  
✓ Proper train/val/test splits with reproducible seeding  
✓ Separate transforms for train vs val/test  
✓ Validates all 10 classes are present  
✓ Prints detailed dataset statistics  

## Notes

- The dataset uses RGB images from Sentinel-2 satellite
- Images are 64x64 pixels (can be resized via config)
- All images are in JPG format
- ImageNet normalization is used (common practice for transfer learning)
