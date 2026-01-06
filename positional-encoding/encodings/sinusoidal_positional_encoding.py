import torch
import math

class SinusoidalPE:
    def __init__(self, d_model: int, seq_len: int) -> None:
        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self) -> torch.Tensor:
        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def __call__(self) -> torch.Tensor:
        return self.forward()
