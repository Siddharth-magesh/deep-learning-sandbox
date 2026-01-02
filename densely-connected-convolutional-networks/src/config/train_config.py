from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class TrainingConfig:
    epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    mixed_precision: bool = True
    grad_clip: Optional[float] = None

    log_interval: int = 50
    save_interval: int = 10

