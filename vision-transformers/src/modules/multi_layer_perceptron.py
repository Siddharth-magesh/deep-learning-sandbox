import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_dim: int, mlp_size: int, dropout: float = 0.1) -> None:
        super(MultiLayerPerceptron, self).__init__()
        self.multilayer_perceptron = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=mlp_size
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=mlp_size,
                out_features=embedding_dim
            ),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multilayer_perceptron(x)