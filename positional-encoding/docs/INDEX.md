# Index Positional Encoding

## Overview

Index Positional Encoding is the simplest possible form of positional encoding. It directly uses the position index (0, 1, 2, ..., n-1) as the encoding, optionally normalized to a [0, 1] range. While rarely used alone in modern transformers, it serves as a building block and baseline for understanding more sophisticated methods.

## Mathematical Foundation

For a sequence of length $n$, the encoding at position $pos$ is simply:

**Non-normalized (raw indices)**:
$$PE_{(pos)} = pos$$

**Normalized**:
$$PE_{(pos)} = \frac{pos}{\max(1, n-1)}$$

The encoding is one-dimensional, directly representing the ordinal position.

## Key Properties

### Ultimate Simplicity
Cannot be simpler - just the position number itself.

### Linear Relationship
The encoding preserves exact linear relationships between positions:
$$PE_{(a)} - PE_{(b)} = a - b$$

### Single Dimension
Unlike other encodings that use multiple dimensions to capture position, index encoding is inherently one-dimensional.

### Unbounded (Raw) or Bounded (Normalized)
- Raw: grows without bound as sequence length increases
- Normalized: always in range [0, 1]

### No Frequency Information
Contains no frequency components or wavelike patterns.

## Implementation Details

**Non-normalized version**:
1. Generate sequence: 0, 1, 2, ..., seq_len-1
2. Reshape to (seq_len, 1) for compatibility

**Normalized version**:
1. Generate sequence: 0, 1, 2, ..., seq_len-1
2. Divide by (seq_len - 1)
3. Result ranges from 0.0 to 1.0

## Advantages

- **Extreme simplicity**: Easiest to understand and implement
- **No computation**: Trivial to generate
- **Exact position preservation**: Maintains precise ordinal information
- **Memory efficient**: Just a range operation
- **Perfect for embeddings**: Can be used as input to learned embedding layers

## Limitations

- **No frequency diversity**: Lacks multi-scale temporal patterns
- **Poor for direct use**: Too simple for most transformer applications
- **Extrapolation issues**: Normalized version fails on longer sequences (values exceed 1)
- **Single dimension**: Cannot capture complex positional relationships
- **Scale sensitivity**: Raw version highly sensitive to sequence length
- **No relative bias**: Requires additional mechanisms to learn relative positions

## Use Cases

Best used as:
- **Input to learned embeddings**: Feed to an embedding layer that learns to map indices to rich representations
- **Baseline for comparison**: Reference point for evaluating other encoding schemes
- **Curriculum learning**: Simple starting point before switching to complex encodings
- **Hybrid approaches**: Combined with other encoding methods
- **Small-scale experiments**: Quick prototyping and debugging

## Comparison to Other Methods

**vs. Sinusoidal**:
- Much simpler but lacks frequency diversity
- Cannot capture multi-scale patterns
- Requires learned transformation for effectiveness

**vs. Learned Embeddings**:
- Index encoding often serves as input to learned embedding layers
- Learned embeddings: maps indices to learned vectors
- Index encoding: provides raw indices for learning

**vs. RoPE/ALiBi**:
- No relative position modeling
- No built-in distance bias
- Requires explicit attention mechanisms to learn relationships

## Extended Use: Learned Positional Embeddings

Index encoding is commonly used as input to learned position embeddings:

```python
position_indices = IndexPE(seq_len=512)()
embedding_layer = nn.Embedding(num_embeddings=512, embedding_dim=768)
learned_positions = embedding_layer(position_indices.long())
```

This approach:
- Learns task-specific position representations
- Can adapt to training data
- Used in BERT and many other models

## Normalization Strategies

**No normalization**: 
- Preserves absolute position values
- Grows with sequence length
- Useful when feeding to embedding layers

**[0, 1] normalization**: 
- Makes positions scale-invariant
- Better for direct use in neural networks
- Prevents exploding values

**[-1, 1] normalization**: 
- Centers around 0
- Can help with numerical stability
- Common in certain architectures

## Example Output Characteristics

Visualization shows:
- Simple linear gradient
- Monotonically increasing values
- No patterns or structure (by design)
- Single column in heatmap (one dimension)
- Perfectly predictable progression
