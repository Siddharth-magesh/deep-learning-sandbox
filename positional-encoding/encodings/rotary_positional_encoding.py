import torch
import math

class RotaryPE:
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        assert d_model % 2 == 0
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        self._compute_freqs()

    def _compute_freqs(self) -> None:
        half = self.d_model // 2
        self.freqs = torch.exp(
            -math.log(self.base) * torch.arange(0, half, dtype=torch.float32) / half
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        if positions is None:
            seq_len = x.shape[-2]
            positions = torch.arange(seq_len, dtype=torch.float32)
        
        half = self.d_model // 2
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)
        sin_val = torch.sin(angles)
        cos_val = torch.cos(angles)
        
        x1 = x[..., :half]
        x2 = x[..., half:]
        
        rotated = torch.cat([
            x1 * cos_val - x2 * sin_val,
            x1 * sin_val + x2 * cos_val
        ], dim=-1)
        
        return rotated

    def __call__(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        return self.forward(x, positions)
