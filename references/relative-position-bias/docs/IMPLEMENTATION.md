# Implementation Details

## Relative Position Indexing

### 1D Indexing Algorithm

The goal is to map pairs of positions to unique relative position indices.

**Step-by-Step Process**:

1. **Create Position Coordinates**:
   ```python
   coords = torch.arange(seq_len)
   ```
   For seq_len=4: [0, 1, 2, 3]

2. **Compute Relative Positions**:
   ```python
   relative_coords = coords[:, None] - coords[None, :]
   ```
   Result matrix:
   ```
   [[ 0, -1, -2, -3],
    [ 1,  0, -1, -2],
    [ 2,  1,  0, -1],
    [ 3,  2,  1,  0]]
   ```

3. **Shift to Non-Negative**:
   ```python
   relative_coords += seq_len - 1
   ```
   Result:
   ```
   [[3, 2, 1, 0],
    [4, 3, 2, 1],
    [5, 4, 3, 2],
    [6, 5, 4, 3]]
   ```

This gives indices from 0 to 2×seq_len-2, which we use to index into the bias table.

### 2D Indexing Algorithm

For 2D positions, we need to handle both height and width dimensions.

**Step-by-Step Process**:

1. **Create Coordinate Grids**:
   ```python
   coords_h = torch.arange(Wh)
   coords_w = torch.arange(Ww)
   coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
   ```
   For window_size=(3, 3):
   - coords[0] (height): [[0,0,0], [1,1,1], [2,2,2]]
   - coords[1] (width): [[0,1,2], [0,1,2], [0,1,2]]

2. **Flatten Coordinates**:
   ```python
   coords_flatten = torch.flatten(coords, 1)
   ```
   Shape: (2, 9) representing 9 positions

3. **Compute Pairwise Relative Coordinates**:
   ```python
   relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
   ```
   Shape: (2, 9, 9) - relative positions for both dimensions

4. **Reshape for Indexing**:
   ```python
   relative_coords = relative_coords.permute(1, 2, 0).contiguous()
   ```
   Shape: (9, 9, 2)

5. **Shift to Non-Negative**:
   ```python
   relative_coords[:, :, 0] += Wh - 1
   relative_coords[:, :, 1] += Ww - 1
   ```

6. **Convert 2D Index to 1D**:
   ```python
   relative_coords[:, :, 0] *= (2 * Ww - 1)
   relative_position_index = relative_coords.sum(-1)
   ```

This converts (rel_h, rel_w) to a single index: `rel_h × (2×Ww-1) + rel_w`

## Bias Table Lookup

The bias table is a learnable parameter matrix:
- Shape: (num_relative_positions, num_heads)
- Initialized with truncated normal distribution

**Lookup Process**:

```python
def forward(self) -> torch.Tensor:
    N = self.relative_position_index.size(0)
    
    bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ]
    
    bias = bias.view(N, N, self.num_heads)
    bias = bias.permute(2, 0, 1).contiguous()
    
    return bias
```

**Steps**:
1. Flatten index matrix: (N, N) → (N²,)
2. Index into bias table: (N²,) → (N², num_heads)
3. Reshape: (N², num_heads) → (N, N, num_heads)
4. Permute: (N, N, num_heads) → (num_heads, N, N)

## Attention Integration

### Standard Attention

```python
attn_scores = torch.matmul(Q, K.transpose(-2, -1))
attn_scores = attn_scores / math.sqrt(head_dim)
attn_weights = F.softmax(attn_scores, dim=-1)
output = torch.matmul(attn_weights, V)
```

### With Relative Position Bias

```python
attn_scores = torch.matmul(Q, K.transpose(-2, -1))
attn_scores = attn_scores / math.sqrt(head_dim)

if relative_position_bias is not None:
    attn_scores = attn_scores + relative_position_bias.unsqueeze(0)

attn_weights = F.softmax(attn_scores, dim=-1)
output = torch.matmul(attn_weights, V)
```

**Key Points**:
- Bias is added before softmax
- Bias shape: (num_heads, N, N)
- Unsqueeze to (1, num_heads, N, N) for broadcasting over batch
- Each head gets its own bias pattern

## Memory Efficiency

### Pre-Computed Indices

We compute indices once during initialization and register as a buffer:

```python
self.register_buffer("relative_position_index", relative_position_index)
```

**Benefits**:
- Computed only once
- Automatically moved to correct device
- Not treated as a parameter (no gradients)
- Saved/loaded with model state

### Shared Bias Across Positions

The same bias is used regardless of absolute position:
- Position (0, 0) to (0, 2) uses same bias as (5, 5) to (5, 7)
- Reduces parameters from O(N²) to O(W²) where W << N

## Numerical Stability

### Initialization

Use truncated normal initialization:

```python
nn.init.trunc_normal_(self.relative_position_bias_table, std=init_std)
```

This prevents extreme initial values that could cause:
- Attention collapse (all attention to one position)
- Gradient explosion
- NaN values

Typical `init_std`: 0.01 to 0.02

### Gradient Flow

Bias is added before softmax, ensuring gradients flow through:

```
Loss → Attention Output → Softmax → Attention Scores → Bias
```

The gradient path is:
```python
∂Loss/∂Bias = ∂Loss/∂Scores × ∂Scores/∂Bias
```

Where `∂Scores/∂Bias = 1` (identity), ensuring stable gradients.

## Shape Transformations

### Multi-Head Attention Pipeline

```
Input: (B, N, embed_dim)
    ↓ QKV Projection
(B, N, 3 × embed_dim)
    ↓ Reshape & Permute
(3, B, num_heads, N, head_dim)
    ↓ Split
Q, K, V: (B, num_heads, N, head_dim)
    ↓ Attention
Scores: (B, num_heads, N, N)
    ↓ Add Bias
Scores + Bias: (B, num_heads, N, N)
    ↓ Softmax & MatMul
Output: (B, num_heads, N, head_dim)
    ↓ Transpose & Reshape
Output: (B, N, embed_dim)
```

### Bias Broadcasting

```
Bias shape:        (num_heads, N, N)
Unsqueezed:        (1, num_heads, N, N)
Scores shape:      (B, num_heads, N, N)
After addition:    (B, num_heads, N, N)
```

Broadcasting happens over the batch dimension.

## Optimization Techniques

### Fused Operations

Combine Q, K, V projections:

```python
self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
```

Instead of:
```python
self.q_proj = nn.Linear(embed_dim, embed_dim)
self.k_proj = nn.Linear(embed_dim, embed_dim)
self.v_proj = nn.Linear(embed_dim, embed_dim)
```

**Benefits**:
- Fewer kernel launches
- Better memory coalescing
- ~1.5x faster

### In-Place Operations

Use in-place operations where possible:

```python
relative_coords[:, :, 0] += Wh - 1
```

Instead of:
```python
relative_coords[:, :, 0] = relative_coords[:, :, 0] + Wh - 1
```

### Contiguous Memory

Ensure contiguous memory after permutations:

```python
relative_coords = relative_coords.permute(1, 2, 0).contiguous()
```

This prevents inefficient strided memory access.

## Edge Cases

### Single Position

For seq_len=1:
- Only one relative position (self)
- Bias table size: 1 × num_heads
- Bias matrix: (num_heads, 1, 1)

### Asymmetric Windows

For window_size=(4, 8):
- Height relative positions: 2×4-1 = 7
- Width relative positions: 2×8-1 = 15
- Total: 7 × 15 = 105 unique positions

### Large Sequences

For very long sequences, consider:
- Windowed attention to reduce memory
- Sparse attention patterns
- Local + global attention hybrid

## Comparison with Alternatives

### T5 Relative Bias

T5 uses scalar biases per relative position:
```python
bias_table: (num_buckets,)
```

Our implementation uses per-head biases:
```python
bias_table: (num_relative_positions, num_heads)
```

**Advantages**:
- More expressive (different heads learn different patterns)
- Better performance on vision tasks

**Disadvantages**:
- More parameters

### Swin Transformer Bias

Swin uses the same approach as our 2D implementation:
- Learnable bias table
- Pre-computed indices
- Per-head biases

Our implementation is compatible with Swin-style architectures.

### ALiBi (Attention with Linear Biases)

ALiBi uses fixed slopes per head:
```python
bias = -slope × distance
```

Our implementation is learnable:
- More flexible
- Can learn non-linear patterns
- Requires more training

## Testing and Validation

### Unit Tests

Test index computation:
```python
def test_1d_indices():
    seq_len = 5
    indices = get_relative_position_index_1d(seq_len)
    assert indices.shape == (seq_len, seq_len)
    assert indices.min() == 0
    assert indices.max() == 2 * seq_len - 2
```

Test bias shape:
```python
def test_bias_shape():
    rpb = RelativePositionBias(num_heads=4, window_size=(7, 7), bias_type="2d")
    bias = rpb()
    assert bias.shape == (4, 49, 49)
```

### Gradient Checks

Verify gradients flow:
```python
rpb = RelativePositionBias(num_heads=4, window_size=(7, 7), bias_type="2d")
bias = rpb()
loss = bias.sum()
loss.backward()
assert rpb.relative_position_bias_table.grad is not None
```
