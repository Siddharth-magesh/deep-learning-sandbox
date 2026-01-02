from dataclasses import dataclass
from typing import Optional

@dataclass
class OptimizerConfig:
    name: str = "adamw"

    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4

    betas: tuple = (0.9, 0.999)

@dataclass
class SchedulerConfig:
    name: Optional[str] = "cosine"

    step_size: int = 30
    gamma: float = 0.1

    t_max: int = 200
