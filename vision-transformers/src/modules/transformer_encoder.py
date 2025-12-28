import torch
import torch.nn as nn
from .multi_layer_perceptron import MultiLayerPerceptron
from .attention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_size: int, attn_dropout: float = 0.1, mlp_dropout: float = 0.1) -> None:
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mha_block = MultiHeadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )
        self.mlp_block = MultiLayerPerceptron(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=mlp_dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha_block(self.layer_norm1(x))
        x = x + self.mlp_block(self.layer_norm2(x))
        return x