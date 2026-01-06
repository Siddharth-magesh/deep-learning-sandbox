import torch

class IndexPE:
    def __init__(self, seq_len: int, normalize: bool = False) -> None:
        self.seq_len = seq_len
        self.normalize = normalize

    def forward(self) -> torch.Tensor:
        indices = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(1)
        if self.normalize:
            indices = indices / max(1, self.seq_len - 1)
        return indices

    def __call__(self) -> torch.Tensor:
        return self.forward()
