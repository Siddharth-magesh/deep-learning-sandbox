from dataclasses import dataclass
from typing import Optional


@dataclass
class OptunaConfig:
    n_trials: int = 50
    timeout: Optional[int] = None
    study_name: str = "gpt2_optimization"
    storage: Optional[str] = None
    direction: str = "minimize"
    sampler: str = "TPE"
    pruner: str = "MedianPruner"
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
