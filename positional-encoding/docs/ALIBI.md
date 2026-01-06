# ALiBi (Attention with Linear Biases)

## Overview

ALiBi (Attention with Linear Biases) was introduced by Press et al. in "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (2021). Instead of adding positional information to embeddings, ALiBi applies a simple linear bias directly to the attention scores based on the distance between query and key positions.

## Mathematical Foundation

For a query at position $i$ attending to a key at position $j$, ALiBi adds a bias to the attention score:

$$\text{score}(i, j) = q_i \cdot k_j + m \cdot (i - j)$$

where:
- $m$ is a head-specific slope (negative value)
- $(i - j)$ is the relative distance between positions
- Different attention heads use different slopes

The slopes for each head are computed as:

$$m_h = 2^{-\frac{8h}{H}}$$

where $h$ is the head index and $H$ is the total number of heads.

## Key Properties

### Linear Penalty by Distance
The bias is simply proportional to the distance between positions. Tokens that are farther apart receive a larger negative bias, naturally encouraging local attention.

### No Positional Embeddings Required
ALiBi completely eliminates the need for positional embeddings in the input, simplifying the model architecture.

### Head-Specific Biases
Each attention head has its own slope, allowing different heads to focus on different ranges:
- Heads with gentle slopes: attend to distant positions
- Heads with steep slopes: focus on nearby positions

### Perfect Extrapolation
ALiBi provides excellent extrapolation to sequences much longer than those seen during training, often with no performance degradation.

## Implementation Details

The implementation involves:

1. Computing slopes for each attention head
2. Creating a relative position matrix (i - j) for all position pairs
3. Multiplying slopes by relative positions to get biases
4. Adding biases to attention scores before softmax

For non-power-of-2 number of heads, a geometric sequence is used to interpolate slopes.

## Advantages

- **Exceptional extrapolation**: Best-in-class performance on longer sequences
- **Simplicity**: Extremely simple to implement and understand
- **No embeddings needed**: Removes positional embeddings entirely
- **Memory efficient**: Only stores per-head slopes, not full position encodings
- **Training efficiency**: Often trains faster than models with learned position embeddings
- **Interpretability**: Linear relationship makes behavior predictable

## Limitations

- **Linear assumption**: Assumes importance decays linearly with distance
- **No absolute position**: Cannot distinguish absolute positions (only relative)
- **Fixed penalty scheme**: Cannot learn task-specific distance penalties
- **Attention modification**: Requires changing attention mechanism implementation

## Use Cases

Ideal for:
- Tasks requiring long-range dependencies
- Applications with varying sequence lengths
- Models that need to extrapolate beyond training lengths
- Scenarios where training efficiency is critical
- Language modeling and text generation
- Any transformer architecture where input length varies significantly

## Comparison with Other Methods

**vs. Sinusoidal PE**: 
- Simpler implementation
- Better extrapolation
- No embedding dimension overhead

**vs. Learned PE**: 
- No parameters to learn
- Guaranteed extrapolation capability
- Faster convergence

**vs. RoPE**: 
- Simpler conceptually
- Better length extrapolation
- Less computational overhead

## Head Slope Distribution

For 8 heads, typical slopes are:
- Head 0: -0.0039 (gentle, long-range)
- Head 1: -0.0078
- Head 2: -0.0156
- ...
- Head 7: -0.5000 (steep, short-range)

This creates a natural hierarchy where different heads specialize in different ranges.

## Example Output Characteristics

The attention bias visualization shows:
- Diagonal pattern (zero bias on diagonal)
- Linear gradients extending from diagonal
- Different intensities for different heads
- Symmetric pattern (distance is symmetric)
- Darker regions for farther positions (higher penalty)
