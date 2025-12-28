from dataclasses import dataclass
import os

@dataclass
class DataConfig:
    dataset_name: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = os.cpu_count() or 4
    pin_memory: bool = True
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2023, 0.1994, 0.2010)
    image_size: int = 32