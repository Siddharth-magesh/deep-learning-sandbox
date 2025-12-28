import torch
from dataclasses import dataclass

@dataclass
class BaseConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"