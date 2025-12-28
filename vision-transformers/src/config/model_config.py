from dataclasses import dataclass

@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    
    embedding_dim: int = 192
    num_heads: int = 3
    num_layers: int = 12
    mlp_size: int = 768
    
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    
    initializer_range: float = 0.02