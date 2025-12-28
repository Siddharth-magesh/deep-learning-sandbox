import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from .dataset import CIFAR10Dataset
from .transform import get_train_transform, get_test_transform

def get_dataloaders(root: str = './data', batch_size: int = 128, num_workers: int = 4, pin_memory: bool = True, val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    full_train_dataset = CIFAR10Dataset(
        root=root,
        train=True,
        transform=get_train_transform()
    )
    
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = CIFAR10Dataset(
        root=root,
        train=False,
        transform=get_test_transform()
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader