import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .relative_position import RelativePositionBias

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale: bool = True, dropout: float = 0.0) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relative_position_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, N, D = query.shape
        attn = torch.matmul(query, key.transpose(-2, -1))
        if self.scale:
            attn = attn / math.sqrt(D)
        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            use_relative_position: bool = False,
            rpb_kwargs: dict | None = None
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_relative_position = use_relative_position
        self.qkv = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim * 3,
            bias=True
        )
        self.attention = ScaledDotProductAttention(
            scale=True,
            dropout=dropout
        )
        if self.use_relative_position:
            assert rpb_kwargs is not None, "rpb_kwargs must be provided when use_relative_position is True"
            self.relative_position_bias = RelativePositionBias(
                **rpb_kwargs
            )
        else:
            self.relative_position_bias = None

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x)  # (B, N, 3*embed_dim)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
            query, key, value = qkv[0], qkv[1], qkv[2]  # each is (B, num_heads, N, head_dim)

            if self.use_relative_position:
                relative_position_bias = self.relative_position_bias()
            else:
                relative_position_bias = None

            out, attn = self.attention(
                query,
                key,
                value,
                relative_position_bias
            )  # out: (B, num_heads, N, head_dim)

            out = out.transpose(1, 2).reshape(B, N, C)
            return out