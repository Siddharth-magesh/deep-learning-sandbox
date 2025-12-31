import torch
import torch.nn as nn

class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(
            4 * dim,
            2 * dim,
            bias=False
        )
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, h, w, C = x.shape
        assert h == H and w == W, f"Input feature has wrong size, expected ({H}, {W}), got ({h}, {w})"
        assert H % 2 == 0 and W % 2 == 0, f"Height and Width must be even, got H: {H}, W: {W}"
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x