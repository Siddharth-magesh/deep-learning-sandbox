# GPT-2 Architecture

## Overview

This implementation follows the GPT-2 architecture as described in "Language Models are Unsupervised Multitask Learners" by OpenAI.

## Model Components

### 1. Token and Position Embeddings

```
Input Shape: (batch_size, sequence_length)
Token Embedding: (vocab_size=50257, d_model=768)
Position Embedding: (context_length=1024, d_model=768)
Output Shape: (batch_size, sequence_length, d_model)
```

The model uses learned positional embeddings that are added to token embeddings.

### 2. Multi-Head Self-Attention

**Parameters:**
- Number of heads: 12
- Head dimension: 64 (d_model / num_heads)
- Total dimension: 768

**Process:**
1. Linear projections for Query, Key, Value
2. Split into multiple heads
3. Scaled dot-product attention with causal masking
4. Concatenate heads
5. Output projection

**Attention Formula:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

**Causal Masking:**
Upper triangular mask prevents attending to future tokens:
```
mask = [[0, -inf, -inf, -inf],
        [0,    0, -inf, -inf],
        [0,    0,    0, -inf],
        [0,    0,    0,    0]]
```

### 3. Feed-Forward Network

```
Layer 1: Linear(d_model=768, d_ff=3072)
Activation: GELU
Dropout: 0.1
Layer 2: Linear(d_ff=3072, d_model=768)
Dropout: 0.1
```

**GELU Activation:**
More effective than ReLU for language modeling tasks.

### 4. Decoder Block

Each transformer block consists of:

```
Input
  |
LayerNorm
  |
Multi-Head Attention
  |
Residual Connection (+)
  |
LayerNorm
  |
Feed-Forward Network
  |
Residual Connection (+)
  |
Output
```

**Key Features:**
- Pre-normalization (LayerNorm before attention/FFN)
- Residual connections for gradient flow
- Dropout for regularization

### 5. Output Layer

```
Final LayerNorm: (d_model=768)
Language Model Head: Linear(d_model=768, vocab_size=50257)
```

**Weight Tying:**
Token embedding weights are shared with the language model head to reduce parameters.

## Complete Architecture

```
GPT2Model(
  Total Layers: 12 decoder blocks
  Total Parameters: ~124M
  
  Components:
  - Token Embedding (50257 × 768)
  - Position Embedding (1024 × 768)
  - 12 × Decoder Block:
      - Multi-Head Attention (12 heads)
      - Feed-Forward Network (768 → 3072 → 768)
      - Layer Normalization (×2)
  - Final Layer Normalization
  - Language Model Head (768 → 50257)
)
```

## Tensor Flow Through Model

### Forward Pass Example

**Input:**
```
input_ids: (batch_size=8, sequence_length=128)
```

**Embedding Layer:**
```
token_embeddings: (8, 128, 768)
position_embeddings: (8, 128, 768)
hidden_states: (8, 128, 768)
```

**Through Decoder Blocks:**
```
Block 1-12: (8, 128, 768) → (8, 128, 768)
```

**Output:**
```
logits: (8, 128, 50257)
```

### Attention Mechanism Detail

**Query, Key, Value Projections:**
```
Input: (batch_size, seq_len, d_model)
       (8, 128, 768)

Q, K, V: (8, 128, 768)

Reshaped to heads:
(8, 12, 128, 64)
```

**Attention Scores:**
```
scores = Q @ K^T / sqrt(64)
Shape: (8, 12, 128, 128)

After masking and softmax:
attention_probs: (8, 12, 128, 128)
```

**Context:**
```
context = attention_probs @ V
Shape: (8, 12, 128, 64)

Concatenated:
Shape: (8, 128, 768)
```

## Initialization Strategy

**Linear Layers:**
- Weight: Normal(mean=0.0, std=0.02)
- Bias: Zeros

**Embeddings:**
- Weight: Normal(mean=0.0, std=0.02)

**Layer Normalization:**
- Weight: Ones
- Bias: Zeros

This initialization ensures stable training from the start.

## Computational Complexity

### Time Complexity per Layer

**Self-Attention:**
- O(n² × d) where n=sequence_length, d=d_model
- Dominant cost for long sequences

**Feed-Forward:**
- O(n × d²) 
- Dominant cost for shorter sequences

### Memory Complexity

**Parameters:**
- Embeddings: ~38.6M parameters
- 12 Blocks: ~85M parameters
- Total: ~124M parameters

**Activations (per sample):**
- Proportional to (num_layers × sequence_length × d_model)
- Increases linearly with sequence length

## Design Choices

### Why Pre-Normalization?

Pre-normalization (LayerNorm before attention/FFN) provides:
- Better gradient flow
- Easier optimization
- More stable training

### Why GELU?

GELU (Gaussian Error Linear Unit) offers:
- Smoother gradients than ReLU
- Better performance on language tasks
- Probabilistic interpretation

### Why Weight Tying?

Sharing weights between token embeddings and output layer:
- Reduces parameters significantly
- Improves generalization
- Common practice in language models

### Causal Masking

Prevents information leakage from future tokens:
- Essential for autoregressive generation
- Ensures proper language modeling objective
- Allows parallel training on sequences

## Comparison with Original GPT-2

| Component | Our Implementation | Original GPT-2 |
|-----------|-------------------|----------------|
| Architecture | Decoder-only | Decoder-only |
| Layers | 12 | 12 (117M) |
| d_model | 768 | 768 |
| Heads | 12 | 12 |
| d_ff | 3072 | 3072 |
| Context | 1024 | 1024 |
| Parameters | ~124M | 117M |
| Vocabulary | 50257 | 50257 |

Our implementation closely matches the GPT-2 (small) configuration.
