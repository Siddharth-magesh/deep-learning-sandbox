from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    max_length: int = 768  # Increased to match model context
    cache_dir: Optional[str] = None
    preprocessing_num_workers: int = 4


@dataclass
class InferenceConfig:
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    do_sample: bool = True
    num_return_sequences: int = 1
    stream: bool = True
