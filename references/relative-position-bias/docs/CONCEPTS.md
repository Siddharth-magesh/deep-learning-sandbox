# Relative Position Bias Concepts

## Overview

Relative position bias is a mechanism to encode positional information in attention-based models without using absolute position embeddings. Instead of learning where each token is in absolute terms, the model learns how tokens relate to each other based on their relative distances.

## Why Relative Position Bias?

### Limitations of Absolute Position Embeddings

Traditional position embeddings assign a unique learned vector to each position in a sequence. This approach has several drawbacks:

1. **Fixed Sequence Length**: Models trained with absolute positions struggle to generalize to longer sequences
2. **No Translation Invariance**: The same pattern at different positions is treated differently
3. **Limited Extrapolation**: Cannot naturally handle positions not seen during training

### Advantages of Relative Position Bias

1. **Length Generalization**: Works with sequences of varying lengths
2. **Translation Invariance**: Relationships are position-agnostic
3. **Parameter Efficiency**: Shares parameters across positions
4. **Better Inductive Bias**: Explicitly models pairwise relationships

## Core Concept

In standard attention, we compute:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

With relative position bias:

```
Attention(Q, K, V) = softmax(QK^T / √d_k + B) V
```

Where B is the relative position bias matrix that depends only on the relative distances between positions, not their absolute locations.

## 1D Relative Position Bias

### Concept

In 1D sequences (like text), the relative position between two positions i and j is simply:

```
relative_position = i - j
```

This can range from -(N-1) to (N-1) where N is the sequence length, giving us 2N-1 possible relative positions.

### Example

For a sequence of length 4:
- Position 0 to Position 2: relative position = 0 - 2 = -2
- Position 3 to Position 1: relative position = 3 - 1 = 2
- Position 2 to Position 2: relative position = 2 - 2 = 0

### Implementation

We maintain a learnable table of size (2N-1, num_heads) and index into it based on the relative position between each pair of tokens.

## 2D Relative Position Bias

### Concept

For 2D data (like images organized in windows), we need to consider both height and width dimensions. For positions (i_h, i_w) and (j_h, j_w):

```
relative_position_h = i_h - j_h
relative_position_w = i_w - j_w
```

Each dimension can range from -(W-1) to (W-1) where W is the window size, giving us (2W_h - 1) × (2W_w - 1) possible relative positions.

### Example

For a 7×7 window:
- Total positions: 49
- Possible relative positions in each dimension: 13 (from -6 to +6)
- Total unique relative positions: 13 × 13 = 169

### Spatial Relationships

2D relative position bias captures:
- Vertical relationships (same column, different rows)
- Horizontal relationships (same row, different columns)
- Diagonal relationships
- Distance-based patterns

## Key Properties

### Learnable Parameters

The bias values are learned during training, allowing the model to discover what spatial relationships are important for the task.

### Shared Across Positions

The same relative position (e.g., "2 positions to the left") uses the same bias regardless of where it occurs in the sequence/image.

### Per-Head Biases

Each attention head can learn different positional preferences, enabling diverse attention patterns.

## Mathematical Details

### Bias Table Indexing

For 2D bias with window size (W_h, W_w):

1. Create coordinate grids for each dimension
2. Compute pairwise relative coordinates
3. Shift to make all values non-negative: add (W-1)
4. Convert 2D index to 1D: `index = rel_h × (2W_w - 1) + rel_w`
5. Look up bias value from learnable table

### Complexity

- **1D Bias**: O(N) parameters where N is sequence length
- **2D Bias**: O(W²) parameters where W is window size
- **Memory**: Constant per window regardless of image size

## Comparison with Absolute Positions

| Aspect | Absolute Position | Relative Position |
|--------|------------------|-------------------|
| Parameter Count | O(max_sequence_length) | O(window_size) |
| Length Generalization | Poor | Good |
| Translation Invariance | No | Yes |
| Local Patterns | Implicit | Explicit |
| Implementation | Simpler | More Complex |

## Applications

### Vision Transformers

- Window-based attention (e.g., Swin Transformer)
- Pyramid architectures
- Object detection and segmentation

### Language Models

- Long-range dependencies
- Document-level understanding
- Structured prediction tasks

### Hybrid Approaches

Some models combine both:
- Absolute positions for global structure
- Relative positions for local relationships

## Initialization

Relative position biases are typically initialized:
- From a normal distribution with small standard deviation (e.g., 0.02)
- Using truncated normal to avoid extreme values
- Zero initialization for baseline comparisons

The learned values often show interesting patterns:
- Stronger bias for nearby positions
- Decay with distance
- Directional preferences (e.g., left vs right)
