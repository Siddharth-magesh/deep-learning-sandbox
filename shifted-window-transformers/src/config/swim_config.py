from dataclasses import dataclass
from typing import List

@dataclass
class SwimConfig:
    image_size: int = 224
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 1000

    embed_dim: int = 96
    depths: List[int] = [2, 2, 6, 2]
    num_heads: List[int] = [3, 6, 12, 24]
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True

    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05