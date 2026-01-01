import torch
import torch.nn as nn

def get_relative_position_index_1d(seq_len: int) -> torch.Tensor:
    coords = torch.arange(seq_len)
    relative_coords = coords[:, None] - coords[None, :]
    relative_coords += seq_len - 1
    return relative_coords

def get_relative_position_index_2d(window_size: tuple[int, int]) -> torch.Tensor:
    Wh, Ww = window_size
    coords_h = torch.arange(Wh)
    coords_w = torch.arange(Ww)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += Wh - 1
    relative_coords[:, :, 1] += Ww - 1
    relative_coords[:, :, 0] *= (2 * Ww - 1)
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, window_size: tuple[int, int] | None = None, seq_len: int | None = None, bias_type: str = "2d", init_std: float = 0.02) -> None:
        super(RelativePositionBias, self).__init__()
        assert bias_type in {"1d", "2d"}, f"Unsupported bias type: {bias_type}"
        self.num_heads = num_heads
        self.bias_type = bias_type

        if bias_type == "1d":
            assert seq_len is not None, "seq_len must be provided for 1d relative position bias"
            self.seq_len = seq_len
            relative_position_index = get_relative_position_index_1d(seq_len)
            self.num_relative_positions = 2 * seq_len - 1
        else:
            assert window_size is not None, "Window size must be provided for 2d relative position bias"
            self.window_size = window_size
            Wh, Ww = window_size
            self.num_relative_positions = (2 * Wh - 1) * (2 * Ww - 1)
            relative_position_index = get_relative_position_index_2d(window_size)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_positions, num_heads)
        )
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=init_std)

    def forward(self) -> torch.Tensor:
        N = self.relative_position_index.size(0)
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        bias = bias.view(N, N, self.num_heads)
        bias = bias.permute(2, 0, 1).contiguous()
        return bias