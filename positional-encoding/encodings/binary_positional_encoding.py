import torch
import math

class BinaryPE:
    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len
        self.bits = max(1, math.ceil(math.log2(max(2, seq_len))))

    def forward(self) -> torch.Tensor:
        pe = torch.zeros(self.seq_len, self.bits)
        for pos in range(self.seq_len):
            binary = bin(pos)[2:].zfill(self.bits)
            pe[pos] = torch.tensor([int(b) for b in binary], dtype=torch.float32)
        return pe

    def __call__(self) -> torch.Tensor:
        return self.forward()
