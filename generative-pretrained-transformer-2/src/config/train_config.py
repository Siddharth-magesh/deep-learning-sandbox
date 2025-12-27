import torch
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_epochs: int = 50
    max_training_hours: float = 12.0  # Maximum training time in hours
    warmup_steps: int = 2000
    gradient_clip: float = 1.0
    accumulation_steps: int = 2
    save_every: int = 500
    eval_every: int = 250
    log_every: int = 50
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    pin_memory: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
