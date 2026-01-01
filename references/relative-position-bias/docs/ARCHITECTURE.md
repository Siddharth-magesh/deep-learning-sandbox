# Architecture Guide

## System Architecture

The relative position bias implementation is organized into modular components that can be used independently or combined in larger models.

## Module Hierarchy

```
RelativePositionBias
    ↓
MultiHeadAttention
    ↓
TransformerBlock
    ↓
VisionTransformer
```

## Core Components

### 1. RelativePositionBias

**Purpose**: Generate position-dependent bias for attention scores

**Key Features**:
- Supports both 1D and 2D bias
- Learnable bias table
- Efficient indexing using pre-computed relative position indices
- Per-head bias values

**Parameters**:
- `num_heads`: Number of attention heads
- `window_size`: (height, width) for 2D bias
- `seq_len`: Sequence length for 1D bias
- `bias_type`: "1d" or "2d"
- `init_std`: Standard deviation for initialization

**Forward Flow**:
1. Index into bias table using relative position indices
2. Reshape to (num_heads, seq_len, seq_len)
3. Return bias matrix

### 2. ScaledDotProductAttention

**Purpose**: Compute attention with optional relative position bias

**Key Features**:
- Scaled attention scores
- Optional dropout
- Bias integration before softmax

**Forward Flow**:
1. Compute attention scores: QK^T
2. Scale by 1/√d_k if enabled
3. Add relative position bias if provided
4. Apply softmax
5. Apply dropout
6. Multiply by values: Attention × V

### 3. MultiHeadAttention

**Purpose**: Multi-head attention mechanism with relative position support

**Components**:
- Single linear layer for Q, K, V projections (efficiency)
- ScaledDotProductAttention for each head
- Optional RelativePositionBias module

**Forward Flow**:
1. Project input to Q, K, V: Linear(x) → [B, N, 3 × embed_dim]
2. Split into Q, K, V and reshape for multi-head
3. Compute relative position bias if enabled
4. Apply attention with bias
5. Concatenate heads
6. Return output

### 4. TransformerBlock

**Purpose**: Standard transformer block with attention and feed-forward network

**Components**:
- LayerNorm (pre-norm architecture)
- MultiHeadAttention
- MLP (feed-forward network)
- Residual connections

**Architecture**:
```
Input
  ↓
LayerNorm → MultiHeadAttention → Add (residual)
  ↓
LayerNorm → MLP → Add (residual)
  ↓
Output
```

**MLP Structure**:
- Linear(embed_dim → mlp_hidden_dim)
- GELU activation
- Dropout
- Linear(mlp_hidden_dim → embed_dim)
- Dropout

### 5. VisionTransformer

**Purpose**: Complete vision transformer with relative position bias support

**Components**:
- PatchEmbedding: Convert image to sequence of patches
- Class token
- Position embeddings (learnable)
- Stack of TransformerBlocks
- Classification head

**Forward Flow**:
1. Patch embedding: [B, C, H, W] → [B, num_patches, embed_dim]
2. Add class token: [B, num_patches+1, embed_dim]
3. Add position embeddings
4. Apply dropout
5. Process through transformer blocks
6. Extract class token
7. Classification head → logits

## Design Decisions

### Pre-Norm vs Post-Norm

Uses **pre-norm** architecture:
- LayerNorm applied before attention/MLP
- More stable training
- Better gradient flow
- Standard in modern transformers

### Single QKV Projection

Uses a single linear layer for all Q, K, V:
- More efficient than three separate layers
- Fewer parameters
- Standard practice in implementations

### Learnable vs Fixed Positions

Combines both approaches:
- Absolute position embeddings: Learnable
- Relative position bias: Learnable
- Can enable/disable each independently

### Bias Integration Point

Bias added before softmax:
- Affects attention distribution
- Scales with attention scores
- Standard approach in literature

## Data Flow Example

### Image Classification

```
Input Image [B, 3, 224, 224]
    ↓
PatchEmbedding [B, 196, 768]
    ↓
Add CLS Token [B, 197, 768]
    ↓
Add Position Embeddings [B, 197, 768]
    ↓
TransformerBlock 1
    ├─ LayerNorm
    ├─ MultiHeadAttention
    │   ├─ Compute Q, K, V
    │   ├─ Get Relative Position Bias [num_heads, 197, 197]
    │   ├─ Attention = softmax(QK^T/√d + Bias)
    │   └─ Output = Attention × V
    └─ MLP
    ↓
... (more blocks)
    ↓
Extract CLS Token [B, 768]
    ↓
Classification Head [B, num_classes]
```

## Window-Based Attention

For efficient processing of large images:

1. Partition image into non-overlapping windows
2. Apply attention within each window
3. Use 2D relative position bias for window size
4. Optional: Shifted windows for cross-window connections

**Benefits**:
- Linear complexity in image size
- Constant bias table size
- Captures local patterns effectively

## Parameter Counts

For typical configuration:
- embed_dim = 96
- num_heads = 4
- window_size = (7, 7)

**Relative Position Bias**:
- Bias table: (2×7-1) × (2×7-1) × 4 = 169 × 4 = 676 parameters

**Multi-Head Attention**:
- QKV projection: 96 × (3 × 96) = 27,648 parameters
- Relative position bias: 676 parameters

**TransformerBlock**:
- Attention: ~27,648 parameters
- MLP: ~73,728 parameters (with mlp_ratio=4)
- LayerNorms: ~384 parameters
- Total: ~101,760 parameters per block

## Computational Complexity

### Attention Complexity
- Standard: O(N²d) where N is sequence length, d is embedding dimension
- With bias: O(N²d + N²H) where H is number of heads
- Bias overhead is typically negligible

### Memory Requirements
- Attention weights: O(BHN²) where B is batch size
- Bias table: O(W²H) where W is window size (constant)
- Activations: O(BNd)

## Extensibility

The architecture is designed for easy extension:

### Adding New Bias Types
1. Implement bias index computation in `relative_position.py`
2. Update `RelativePositionBias.__init__` to handle new type
3. No changes needed in attention or transformer

### Custom Attention Patterns
1. Extend `ScaledDotProductAttention`
2. Override forward method
3. Maintain same interface

### Different Position Encodings
1. Modify `VisionTransformer.pos_embed`
2. Or disable and rely only on relative bias
3. Can combine multiple encoding schemes
