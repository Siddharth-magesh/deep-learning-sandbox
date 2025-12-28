import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, root: str = './data', train: bool = True, transform: transforms.Compose = None) -> None:
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
        self.train = train
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        image, label = self.dataset[idx]
        return {
            'image': image,
            'label': label
        }
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def classes(self) -> list:
        return self.dataset.classes