import torch
from torch.utils.data import Dataset
import numpy as np


class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples: int, img_size: int, patch_size: int, in_channels: int = 3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.randn(self.in_channels, self.img_size, self.img_size)
        label = torch.randint(0, 10, (1,)).item()
        return image, label


class PatchEmbedding(torch.nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
