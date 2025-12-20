import os
import kagglehub
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def download_eurosat_dataset(config):
    print("=" * 60)
    print("DOWNLOADING EUROSAT DATASET FROM KAGGLE")
    print("=" * 60)
    
    try:
        dataset_path = kagglehub.dataset_download(config.dataset_name)
        print(f"Dataset downloaded to: {dataset_path}")
        
        dataset_dir = Path(dataset_path)
        eurosat_dir = dataset_dir / "EuroSAT"
        
        if not eurosat_dir.exists():
            subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            for subdir in subdirs:
                nested_eurosat = subdir / "EuroSAT"
                if nested_eurosat.exists():
                    eurosat_dir = nested_eurosat
                    break
        
        if not eurosat_dir.exists():
            raise FileNotFoundError(f"EuroSAT directory not found in: {dataset_path}")
        
        print(f"✓ EuroSAT directory: {eurosat_dir}")
        print("=" * 60)
        
        return eurosat_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def get_transforms(config):
    train_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform


def create_data_loaders(config):
    print("=" * 60)
    print("CREATING DATASETS AND DATA LOADERS")
    print("=" * 60)
    
    eurosat_dir = download_eurosat_dataset(config)
    
    train_transform, val_test_transform = get_transforms(config)
    
    full_dataset = datasets.ImageFolder(root=eurosat_dir, transform=train_transform)
    
    total_size = len(full_dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    val_dataset.dataset = datasets.ImageFolder(root=eurosat_dir, transform=val_test_transform)
    test_dataset.dataset = datasets.ImageFolder(root=eurosat_dir, transform=val_test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.number_of_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.number_of_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.number_of_workers,
        pin_memory=config.pin_memory
    )
    
    print(f"✓ Train samples: {train_size}")
    print(f"✓ Validation samples: {val_size}")
    print(f"✓ Test samples: {test_size}")
    print(f"✓ Classes: {full_dataset.classes}")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader, full_dataset.classes
