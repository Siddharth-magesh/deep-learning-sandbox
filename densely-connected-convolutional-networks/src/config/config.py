from dataclasses import dataclass
from dataclasses import field
from .model_config import DenseNetConfig
from .data_config import DataConfig
from .optim_config import OptimizerConfig, SchedulerConfig
from .train_config import TrainingConfig
from .runtime_config import RuntimeConfig

@dataclass
class Config:
    model: DenseNetConfig = field(default_factory=DenseNetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
