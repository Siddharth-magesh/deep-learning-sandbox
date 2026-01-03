import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_transforms(cfg, is_train: bool = True):
    dataset = cfg.data.dataset.lower()
    
    if dataset in ["cifar10", "cifar100"]:
        if is_train and cfg.data.random_crop:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
            ]
        else:
            transform_list = []
        
        if is_train and cfg.data.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        transform_list.append(transforms.ToTensor())
        
        if cfg.data.normalize:
            if dataset == "cifar10":
                transform_list.append(
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010]
                    )
                )
            else:
                transform_list.append(
                    transforms.Normalize(
                        mean=[0.5071, 0.4867, 0.4408],
                        std=[0.2675, 0.2565, 0.2761]
                    )
                )
    
    elif dataset == "imagenet":
        if is_train:
            transform_list = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transform_list = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        
        if cfg.data.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return transforms.Compose(transform_list)


def build_dataset(cfg, is_train: bool = True):
    dataset_name = cfg.data.dataset.lower()
    transform = get_transforms(cfg, is_train=is_train)
    
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=cfg.data.data_dir,
            train=is_train,
            transform=transform,
            download=True
        )
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=cfg.data.data_dir,
            train=is_train,
            transform=transform,
            download=True
        )
    elif dataset_name == "imagenet":
        split = "train" if is_train else "val"
        dataset = datasets.ImageNet(
            root=cfg.data.data_dir,
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset


def build_dataloaders(cfg) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_dataset = build_dataset(cfg, is_train=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True
    )

    val_dataset = build_dataset(cfg, is_train=False)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def build_test_loader(cfg) -> DataLoader:
    test_dataset = build_dataset(cfg, is_train=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    return test_loader
