import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embedding_dim: int) -> None:
        super(PatchEmbedding, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patch_embedder = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
            padding=0
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        x = x + self.position_embeddings
        return x