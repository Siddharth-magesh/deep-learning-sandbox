import torch
import torch.nn as nn

class MultiHeadPerceptronClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super(MultiHeadPerceptronClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(
                in_features=embedding_dim,
                out_features=num_classes
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)