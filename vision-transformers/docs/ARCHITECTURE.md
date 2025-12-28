# Vision Transformer Architecture

## Overview

The Vision Transformer (ViT) applies the transformer architecture directly to images by treating an image as a sequence of patches.

## Model Components

### 1. Patch Embedding

Converts input image into sequence of patch embeddings.

**Process:**
1. Split image into fixed-size patches (4x4)
2. Flatten each patch
3. Linear projection to embedding dimension
4. Add learnable position embeddings
5. Prepend classification token

**Input:** (batch_size, 3, 32, 32)
**Output:** (batch_size, 65, 192)
- 64 patches + 1 CLS token
- 192 embedding dimension

### 2. Transformer Encoder

Stack of transformer encoder blocks with:
- Multi-head self-attention
- Layer normalization (pre-norm)
- MLP with GELU activation
- Residual connections

**Architecture per block:**
```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

### 3. Multi-Head Self-Attention

**Computation:**
```
Q = x @ Wq
K = x @ Wk
V = x @ Wv

Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**Parameters:**
- Embedding dimension: 192
- Number of heads: 3
- Head dimension: 64
- Attention dropout: 0.1

### 4. MLP Block

Feed-forward network with:
- Linear layer (192 -> 768)
- GELU activation
- Dropout (0.1)
- Linear layer (768 -> 192)
- Dropout (0.1)

**Expansion ratio:** 4x (768 / 192 = 4)

### 5. Classification Head

Final classification layer:
- Layer normalization
- Linear layer (192 -> 10)

Uses CLS token output for classification.

## Model Variants

### Small (Default)
- Embedding dim: 192
- Layers: 12
- Heads: 3
- Parameters: ~2.7M

### Base
- Embedding dim: 384
- Layers: 12
- Heads: 6
- Parameters: ~11M

### Large
- Embedding dim: 512
- Layers: 24
- Heads: 8
- Parameters: ~42M

## Forward Pass

```
1. Patch Embedding
   Input: (B, 3, 32, 32)
   -> Conv2d(3, 192, kernel=4, stride=4)
   -> Flatten + Transpose
   -> Add CLS token
   -> Add position embeddings
   Output: (B, 65, 192)

2. Transformer Encoders (x12)
   For each layer:
      -> LayerNorm
      -> MultiHeadAttention
      -> Residual connection
      -> LayerNorm
      -> MLP
      -> Residual connection
   Output: (B, 65, 192)

3. Classification
   -> Extract CLS token: (B, 192)
   -> LayerNorm
   -> Linear(192, 10)
   Output: (B, 10)
```

## Parameter Distribution

Total Parameters: ~2.7M

- Patch Embedding: ~10K
  - Convolution: 9,216
  - Position embeddings: 12,480
  
- Transformer Encoders: ~2.6M
  - Attention layers: ~1.3M
  - MLP layers: ~1.2M
  - Layer norms: ~50K
  
- Classification Head: ~2K
  - Layer norm: 384
  - Linear: 1,930

## Key Design Choices

### Pre-norm vs Post-norm
Uses pre-norm (layer norm before attention/MLP):
- More stable training
- Better gradient flow
- Faster convergence

### CLS Token
Special classification token:
- Aggregates information from all patches
- Used for final classification
- Learnable embedding

### Positional Encoding
Learnable position embeddings:
- More flexible than sinusoidal
- Can learn 2D spatial relationships
- Better for fixed image sizes

### Patch Size
4x4 patches for 32x32 images:
- 64 patches total
- Balances sequence length vs patch resolution
- Computationally efficient

## Computational Complexity

Attention complexity: O(n^2 * d)
- n = 65 (number of tokens)
- d = 192 (embedding dimension)

MLP complexity: O(n * d * 4d)

Total FLOPs per forward pass: ~40M

## Comparison with CNNs

**Advantages:**
- Global receptive field from layer 1
- Better scalability with data
- Interpretable attention patterns
- Transfer learning friendly

**Disadvantages:**
- Requires more data to train
- Higher memory usage
- Slower inference than CNNs

## Implementation Details

### Weight Initialization
- Linear layers: Normal(0, 0.02)
- Conv layers: Kaiming normal
- Layer norms: Ones for weight, zeros for bias

### Attention Mechanism
- Scaled dot-product attention
- No causal masking (bidirectional)
- Dropout after softmax

### Residual Connections
All residual connections use direct addition without any transformation.
