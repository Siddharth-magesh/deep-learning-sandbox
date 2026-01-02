from dataclasses import dataclass
import os

@dataclass
class DataConfig:
    dataset: str = "imagenet"  
    data_dir: str = "./data"

    batch_size: int = 32
    num_workers: int = os.cpu_count() or 2
    pin_memory: bool = True

    random_crop: bool = True
    random_flip: bool = True
    normalize: bool = True
