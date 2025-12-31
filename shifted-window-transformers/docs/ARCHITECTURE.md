# Swin Transformer Architecture

## Overview

The Swin Transformer constructs hierarchical feature maps by starting from small-sized patches and gradually merging neighboring patches in deeper layers. It computes self-attention within local windows, achieving linear complexity.

## Architecture Diagram

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────┐
│   Patch Embedding   │  4×4 patches → 56×56×96
│   (Patch Partition  │
│    + Linear Embed)  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│      Stage 1        │  56×56×96
│  (2 Swin Blocks)    │  W-MSA → SW-MSA
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Patch Merging     │  56×56×96 → 28×28×192
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│      Stage 2        │  28×28×192
│  (2 Swin Blocks)    │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Patch Merging     │  28×28×192 → 14×14×384
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│      Stage 3        │  14×14×384
│  (6 Swin Blocks)    │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Patch Merging     │  14×14×384 → 7×7×768
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│      Stage 4        │  7×7×768
│  (2 Swin Blocks)    │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Global Avg Pool    │  7×7×768 → 768
│  + Classification   │  768 → num_classes
└─────────────────────┘
```

## Core Components

### 1. Patch Embedding

Splits the image into non-overlapping patches and projects them to embedding dimension.

```python
# Input: (B, 3, 224, 224)
# Patch size: 4×4
# Output: (B, 56, 56, 96)

self.proj = nn.Conv2d(
    in_channels=3,
    out_channels=96,
    kernel_size=4,
    stride=4
)
```

**Computation:**
- Number of patches: (224/4) × (224/4) = 56 × 56 = 3136
- Each patch: 4 × 4 × 3 = 48 pixels → projected to 96 dimensions

### 2. Window Partition

Divides feature map into non-overlapping windows for local attention.

```python
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: int (typically 7)
    Returns:
        windows: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, 
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows
```

**Example (Stage 1):**
- Input: (B, 56, 56, 96)
- Window size: 7
- Number of windows: (56/7) × (56/7) = 64
- Output: (B×64, 7, 7, 96)

### 3. Window Attention (W-MSA)

Standard multi-head self-attention computed within each window.

```
Attention(Q, K, V) = SoftMax(QK^T / √d + B) V
```

Where B is the relative position bias.

**Relative Position Bias:**
- Parameterized bias table: (2M-1) × (2M-1) × num_heads
- M = window_size = 7
- Table size: 13 × 13 × num_heads

```python
# Relative position index computation
coords_h = torch.arange(window_size)  # [0, 1, ..., 6]
coords_w = torch.arange(window_size)  # [0, 1, ..., 6]
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, 7, 7
coords_flatten = coords.flatten(1)  # 2, 49

# Relative coords: coords_flatten[:, :, None] - coords_flatten[:, None, :]
# Shape: 2, 49, 49
# Values range: [-(M-1), M-1] → shift to [0, 2M-2]
```

### 4. Shifted Window Attention (SW-MSA)

Shifts the feature map before partitioning to enable cross-window connections.

```
Regular Windows:          Shifted Windows (shift = window_size // 2 = 3):

┌───┬───┬───┬───┐         ┌─┬─────┬─────┬───┐
│ 0 │ 1 │ 2 │ 3 │         │A│  B  │  C  │ D │
├───┼───┼───┼───┤         ├─┼─────┼─────┼───┤
│ 4 │ 5 │ 6 │ 7 │   →     │E│  F  │  G  │ H │
├───┼───┼───┼───┤         ├─┼─────┼─────┼───┤
│ 8 │ 9 │10 │11 │         │I│  J  │  K  │ L │
├───┼───┼───┼───┤         ├─┼─────┼─────┼───┤
│12 │13 │14 │15 │         │M│  N  │  O  │ P │
└───┴───┴───┴───┘         └─┴─────┴─────┴───┘
```

**Cyclic Shift:**
```python
if self.shift_size > 0:
    x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
```

**Attention Mask:**
To prevent attention between non-adjacent regions after cyclic shift:
```python
# Mask computation ensures tokens from different original windows
# don't attend to each other
attn_mask[region_i != region_j] = -100.0  # Large negative value
```

### 5. Swin Transformer Block

Each block contains:
1. LayerNorm
2. Window/Shifted-Window Multi-head Self-Attention
3. Residual connection
4. LayerNorm
5. MLP (2-layer with GELU)
6. Residual connection

```python
# Forward pass
def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    
    # Cyclic shift (for SW-MSA)
    if self.shift_size > 0:
        x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    
    # Window partition
    x_windows = window_partition(x, self.window_size)
    
    # Attention
    attn_windows = self.attn(x_windows, mask=self.attn_mask)
    
    # Reverse window partition
    x = window_reverse(attn_windows, self.window_size, H, W)
    
    # Reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    
    # Residual + MLP
    x = shortcut + x
    x = x + self.mlp(self.norm2(x))
    return x
```

### 6. Patch Merging (Downsampling)

Reduces spatial resolution by 2× and doubles channel dimension.

```python
def forward(self, x):
    # x: (B, H, W, C)
    # Concatenate 2×2 neighboring patches
    x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
    
    x = self.norm(x)
    x = self.reduction(x)  # Linear: 4C → 2C
    return x  # (B, H/2, W/2, 2C)
```

## Computational Complexity

### Standard ViT (Global Attention)
```
Ω(MSA) = 4hwC² + 2(hw)²C
```
- Quadratic complexity with image size

### Swin Transformer (Window Attention)
```
Ω(W-MSA) = 4hwC² + 2M²hwC
```
- Linear complexity with image size (M is fixed window size)

**Example:**
- Image: 224×224, Patch: 4×4 → h×w = 56×56 = 3136
- Window: 7×7 → M² = 49
- Swin saves: 2×3136²×C vs 2×49×3136×C = **64× fewer operations**

## Feature Map Resolutions

| Stage | Resolution | Channels | Tokens |
|-------|------------|----------|--------|
| Patch Embed | 56×56 | 96 | 3136 |
| Stage 1 | 56×56 | 96 | 3136 |
| Stage 2 | 28×28 | 192 | 784 |
| Stage 3 | 14×14 | 384 | 196 |
| Stage 4 | 7×7 | 768 | 49 |

## Configuration (Swin-Tiny)

```python
SwimConfig(
    image_size=224,
    patch_size=4,
    in_channels=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],      # Blocks per stage
    num_heads=[3, 6, 12, 24], # Heads per stage
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1
)
```

## References

1. [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
2. [Official Implementation](https://github.com/microsoft/Swin-Transformer)
3. [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
