import torch
import torch.nn as nn
from .window_ops import window_partition, window_reverse
from .attention import WindowAttention
from .mlp import MLP

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        H, W = input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"
        if min(H, W) <= window_size:
            self.window_size = min(H, W)
            self.shift_size = 0

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_dim,
            drop=drop
        )

        if self.shift_size > 0:
            self.register_buffer(
                "attn_mask",
                self._create_attn_mask(H, W)
            )
        else:
            self.attn_mask = None

    def _create_attn_mask(self, H: int, W: int) -> torch.Tensor:
        img_mask = torch.zeros((1, H, W, 1))
        cnt = 0
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(
            -1,
            self.window_size * self.window_size
        )

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, _, _, C = x.shape
        assert C == self.dim, "Input feature dimension must match layer dimension"
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            C
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            C
        )
        x = window_reverse(
            attn_windows,
            self.window_size,
            H,
            W
        )
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x