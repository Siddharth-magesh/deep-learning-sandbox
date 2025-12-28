from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_epochs: int = 100
    max_training_hours: float = 24.0
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    save_every: int = 5
    log_every: int = 50
    eval_every: int = 1
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    label_smoothing: float = 0.1