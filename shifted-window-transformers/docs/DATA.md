# Data Loading Guide

Documentation for dataset handling and data loading.

## Supported Datasets

### Tiny-ImageNet (Default)

A subset of ImageNet with 200 classes.

| Property | Value |
|----------|-------|
| Classes | 200 |
| Training images | 100,000 (500 per class) |
| Validation images | 10,000 (50 per class) |
| Image size | 64×64 (resized to 224×224) |
| Source | HuggingFace: `zh-plus/tiny-imagenet` |

## Data Pipeline

```
┌─────────────────┐
│  HuggingFace    │
│    Dataset      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TinyImageNet   │
│    Wrapper      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Transforms    │
│  (Augmentation) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DataLoader    │
│  (Batching)     │
└─────────────────┘
```

## Usage

### Basic Loading

```python
from shifted_window_transformers.src.data import (
    TinyImageNetDataset,
    get_dataloader,
    train_transformation,
    test_transformation
)
from shifted_window_transformers.src.config import DataConfig

# Configuration
config = DataConfig(
    batch_size=64,
    image_size=224,
    num_workers=4
)

# Create transforms
train_transform = train_transformation(
    mean=config.mean,
    std=config.std,
    image_size=config.image_size
)

test_transform = test_transformation(
    mean=config.mean,
    std=config.std,
    image_size=config.image_size
)

# Load dataset
dataset = TinyImageNetDataset(transform=train_transform)
train_dataset, val_dataset = dataset.get_dataset_splits()

# Create data loaders
train_loader, val_loader = get_dataloader(
    train_dataset=train_dataset,
    test_dataset=val_dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory
)

# Iterate
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")  # (64, 3, 224, 224)
    print(f"Labels shape: {labels.shape}")  # (64,)
    break
```

## Transforms

### Training Transforms

```python
def train_transformation(mean, std, image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
```

### Test Transforms

```python
def test_transformation(mean, std, image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
```

### Advanced Augmentation

For better performance, consider adding:

```python
from torchvision import transforms

def advanced_train_transform(mean, std, image_size):
    return transforms.Compose([
        # Resize with random crop
        transforms.Resize(int(image_size * 1.14)),  # 256 for 224
        transforms.RandomCrop(image_size),
        
        # Flipping
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Color augmentation
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        
        # RandAugment (requires torchvision >= 0.11)
        transforms.RandAugment(num_ops=2, magnitude=9),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize
        transforms.Normalize(mean=mean, std=std),
        
        # Random erasing
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33))
    ])
```

## Normalization Statistics

### Tiny-ImageNet

```python
mean = (0.4802, 0.4481, 0.3975)
std = (0.2302, 0.2265, 0.2262)
```

### ImageNet

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

### Computing Custom Statistics

```python
def compute_mean_std(dataset):
    """Compute mean and std of a dataset."""
    loader = DataLoader(dataset, batch_size=100, num_workers=4)
    
    mean = 0.0
    std = 0.0
    n_samples = 0
    
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_size
    
    mean /= n_samples
    std /= n_samples
    
    return mean.tolist(), std.tolist()
```

## Custom Datasets

### Using ImageFolder

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# For datasets organized as:
# data/
#   train/
#     class1/
#       img1.jpg
#       img2.jpg
#     class2/
#       ...
#   val/
#     class1/
#       ...

train_dataset = ImageFolder(
    root="data/train",
    transform=train_transform
)

val_dataset = ImageFolder(
    root="data/val", 
    transform=test_transform
)
```

### Using CIFAR-10

```python
from torchvision.datasets import CIFAR10

# CIFAR-10 normalization
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.Resize(224),  # Upsample to 224×224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std)
])

train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)
```

### Creating a Custom Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        for class_idx, class_name in enumerate(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_idx))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

## DataLoader Configuration

### Recommended Settings

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,               # Shuffle training data
    num_workers=4,              # Parallel data loading
    pin_memory=True,            # Faster GPU transfer
    drop_last=True,             # Drop incomplete batches
    persistent_workers=True     # Keep workers alive
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=128,             # Can use larger batch for eval
    shuffle=False,              # Don't shuffle validation
    num_workers=4,
    pin_memory=True,
    drop_last=False             # Keep all samples for eval
)
```

### Memory Optimization

For large datasets or limited memory:

```python
# Use fewer workers
num_workers = 2

# Reduce prefetch factor
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=2,
    prefetch_factor=2  # Default is 2
)
```

## Debugging Data Loading

### Visualize Samples

```python
import matplotlib.pyplot as plt

def visualize_batch(images, labels, mean, std, class_names=None):
    """Visualize a batch of images."""
    # Denormalize
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    images = images * std + mean
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].permute(1, 2, 0).numpy()
            img = img.clip(0, 1)
            ax.imshow(img)
            label = class_names[labels[i]] if class_names else labels[i].item()
            ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Usage
images, labels = next(iter(train_loader))
visualize_batch(images[:8], labels[:8], config.mean, config.std)
```

### Check Data Statistics

```python
def check_data_stats(loader, num_batches=10):
    """Check mean and std of loaded data."""
    all_means = []
    all_stds = []
    
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        all_means.append(images.mean(dim=(0, 2, 3)))
        all_stds.append(images.std(dim=(0, 2, 3)))
    
    mean = torch.stack(all_means).mean(0)
    std = torch.stack(all_stds).mean(0)
    
    print(f"Data mean: {mean.tolist()}")
    print(f"Data std: {std.tolist()}")
    print("(Should be close to [0,0,0] and [1,1,1] after normalization)")

check_data_stats(train_loader)
```
