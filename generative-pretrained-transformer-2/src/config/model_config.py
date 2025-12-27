from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    context_length: int = 768  # Increased from 512
    d_model: int = 640  # Increased from 512
    num_heads: int = 10  # Increased from 8
    num_layers: int = 8  # Increased from 6
    d_ff: int = 2560  # Increased from 2048 (4x d_model)
    dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_bias: bool = True
