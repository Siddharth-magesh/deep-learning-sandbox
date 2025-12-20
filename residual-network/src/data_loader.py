import os
import kagglehub
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Subset
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
        
        class_dirs = [d for d in eurosat_dir.iterdir() if d.is_dir()]
        print(f"✓ EuroSAT directory: {eurosat_dir}")
        print(f"✓ Found {len(class_dirs)} classes:")
        for cls_dir in sorted(class_dirs):
            num_files = len(list(cls_dir.glob('*.jpg')))
            print(f"    - {cls_dir.name}: {num_files} images")
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
    
    full_dataset = datasets.ImageFolder(root=str(eurosat_dir))
    
    total_size = len(full_dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset split:")
    print(f"  Total images: {total_size}")
    print(f"  Train: {train_size} ({config.train_split*100:.0f}%)")
    print(f"  Validation: {val_size} ({config.val_split*100:.0f}%)")
    print(f"  Test: {test_size} ({config.test_split*100:.0f}%)")
    
    generator = torch.Generator().manual_seed(config.seed)
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size, test_size], generator=generator
    )
    
    train_dataset = datasets.ImageFolder(root=str(eurosat_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(eurosat_dir), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=str(eurosat_dir), transform=val_test_transform)
    
    train_dataset = Subset(train_dataset, train_indices.indices)
    val_dataset = Subset(val_dataset, val_indices.indices)
    test_dataset = Subset(test_dataset, test_indices.indices)
    
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
    
    class_names = full_dataset.classes
    print(f"\n✓ Classes ({len(class_names)}): {class_names}")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader, class_names
