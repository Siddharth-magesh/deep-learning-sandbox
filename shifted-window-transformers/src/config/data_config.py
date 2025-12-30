from dataclasses import dataclass
import os

@dataclass
class DataConfig:
    dataset_name: str = "tiny-imagenet"
    data_dir: str = "./data/tiny-imagenet"
    batch_size: int = 128
    num_workers: int = os.cpu_count() or 2
    pin_memory: bool = True
    mean: tuple = (0.4802, 0.4481, 0.3975)
    std: tuple = (0.2302, 0.2265, 0.2262)
    image_size: int = 224