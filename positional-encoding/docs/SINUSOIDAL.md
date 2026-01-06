# Sinusoidal Positional Encoding

## Overview

Sinusoidal Positional Encoding is the original positional encoding method introduced in the seminal "Attention is All You Need" paper by Vaswani et al. (2017). It provides a deterministic way to inject position information into transformer models using sine and cosine functions of different frequencies.

## Mathematical Foundation

The encoding for position $pos$ and dimension $i$ is defined as:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

where:
- $pos$ is the position in the sequence (0 to seq_len-1)
- $i$ is the dimension index (0 to d_model/2-1)
- $d_{model}$ is the embedding dimension

## Key Properties

### Wavelength Progression
Each dimension corresponds to a sinusoid with wavelength forming a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. This creates a spectrum of frequencies that can capture both fine-grained local patterns and long-range dependencies.

### Relative Position Learning
The encoding enables the model to learn relative positions because for any fixed offset $k$:

$$PE_{pos+k} = f(PE_{pos})$$

This is a linear function of $PE_{pos}$, allowing the model to easily learn attention patterns based on relative distances.

### Bounded Values
All values are bounded between -1 and 1, ensuring numerical stability during training and preventing gradient explosion.

### Deterministic and Fixed
The encoding is completely deterministic and doesn't require learning. It remains fixed throughout training, which means it generalizes well to sequences of different lengths.

## Implementation Details

The implementation follows these steps:

1. Create a zero matrix of shape (seq_len, d_model)
2. Generate position indices from 0 to seq_len-1
3. Compute the division term: $\exp\left(-\frac{2i \cdot \log(10000)}{d_{model}}\right)$
4. Apply sine to even indices
5. Apply cosine to odd indices

## Advantages

- **No learnable parameters**: Reduces model complexity
- **Extrapolation capability**: Can handle sequences longer than those seen during training
- **Smooth transitions**: Continuous functions provide smooth positional information
- **Efficient computation**: Can be precomputed and cached

## Limitations

- **Fixed pattern**: Cannot adapt to specific tasks or data distributions
- **Dimension coupling**: Even and odd dimensions are coupled through the same frequency
- **Absolute position focus**: Primarily encodes absolute positions, though relative positions are learnable

## Use Cases

Best suited for:
- Standard sequence modeling tasks
- Models requiring extrapolation to longer sequences
- Applications where interpretability is important
- Scenarios with limited computational resources (no additional parameters to learn)

## Example Output Characteristics

The heatmap visualization typically shows:
- Vertical stripes representing different frequency components
- Gradual color transitions in lower dimensions (low frequency)
- Rapid oscillations in higher dimensions (high frequency)
- Symmetric patterns due to sine/cosine relationship
