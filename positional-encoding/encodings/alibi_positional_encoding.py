import torch

class ALiBiPE:
    def __init__(self, num_heads: int, max_seq_len: int = 2048) -> None:
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.slopes = self._get_slopes()

    def _get_slopes(self) -> torch.Tensor:
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(torch.arange(n, dtype=torch.float32).log2() - torch.tensor(n).log2()))
            return start

        if self.num_heads & (self.num_heads - 1) == 0:
            slopes = get_slopes_power_of_2(self.num_heads)
        else:
            closest_power_of_2 = 2 ** int(torch.tensor(self.num_heads).log2().floor())
            slopes_1 = get_slopes_power_of_2(closest_power_of_2)
            slopes_2 = self._get_slopes_from_closest(2 * closest_power_of_2)
            slopes_2 = slopes_2[0::2][:self.num_heads - closest_power_of_2]
            slopes = torch.cat([slopes_1, slopes_2])
        
        return slopes

    def _get_slopes_from_closest(self, n: int) -> torch.Tensor:
        ratio = 2 ** (-8.0 / n)
        return ratio ** torch.arange(1, n + 1, dtype=torch.float32)

    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=torch.float32)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        bias = self.slopes.unsqueeze(-1).unsqueeze(-1) * relative_positions.unsqueeze(0)
        
        return bias

    def __call__(self, seq_len: int) -> torch.Tensor:
        return self.forward(seq_len)
