import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    dataset_name: str = "apollo2506/eurosat-dataset"
    data_dir: Path = None
    num_classes: int = 10
    img_size: int = 64

    model_name: str = "resnet50"
    pretrained: bool = False

    batch_size: int = 8 #64
    num_epochs: int = 5 #100
    learning_rate: float = 0.001 #look for better learning rate
    weight_decay: float = 1e-4

    momentum: float = 0.9
    number_of_workers: int = 4 #8

    pin_memory: bool = False #True
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    use_amp: bool = False #True while on GPU
    use_scheduler: bool = False #True while on GPU

    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1

    save_dir: Path = Path("residual-network/checkpoints")
    save_best_only: bool = True
    
    tensorboard_dir: Path = None
    profiler_dir: Path = Path("residual-network/profiler_logs")

    log_interval: int = 10
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = Path.home() / ".cache" / "eurosat-dataset"
        if self.tensorboard_dir is None:
            self.tensorboard_dir = Path(f"residual-network/runs/{self.model_name}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.profiler_dir.mkdir(parents=True, exist_ok=True)

    def display(self):
        print("Configuration:")
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            print(f"  {field}: {value}")