# Positional Encoding

A comprehensive implementation and visualization suite for various positional encoding methods used in transformer architectures and sequence modeling.

## Overview

Positional encodings are essential components of transformer models, providing information about the position of tokens in a sequence. This project implements and visualizes five different positional encoding strategies, each with unique properties and use cases.

## Implemented Encodings

### 1. Sinusoidal Positional Encoding
The original method from "Attention is All You Need" (Vaswani et al., 2017). Uses sine and cosine functions of different frequencies to create a deterministic encoding that captures both absolute and relative positions.

**Key Features:**
- No learnable parameters
- Smooth transitions between positions
- Good extrapolation to longer sequences
- Multi-scale frequency representation

### 2. Rotary Positional Encoding (RoPE)
Introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021). Applies rotation transformations to embedding pairs based on position.

**Key Features:**
- Encodes relative positions naturally
- Excellent length extrapolation
- Used in modern LLMs (LLaMA, PaLM)
- Integrates directly into attention mechanism

### 3. ALiBi (Attention with Linear Biases)
From "Train Short, Test Long" (Press et al., 2021). Adds linear biases to attention scores based on key-query distance.

**Key Features:**
- Simplest implementation
- Best-in-class extrapolation
- No positional embeddings needed
- Head-specific distance penalties

### 4. Binary Positional Encoding
Represents positions using binary bit patterns. Ultra-compact with logarithmic dimensionality.

**Key Features:**
- Minimal memory footprint
- Hierarchical bit structure
- Discrete representation
- Useful for extremely long sequences

### 5. Index Positional Encoding
Direct use of position indices, optionally normalized. Serves as a baseline and input for learned embeddings.

**Key Features:**
- Extreme simplicity
- Often used with learned embeddings
- Perfect position preservation
- Good debugging tool

## Installation

```bash
uv add torch matplotlib seaborn numpy
```

## Project Structure

```
positional-encoding/
├── main.py                 # Demonstration script
├── config/
│   └── __init__.py        # Configuration parameters
├── encodings/
│   ├── __init__.py
│   ├── sinusoidal_positional_encoding.py
│   ├── rotary_positional_encoding.py
│   ├── alibi_positional_encoding.py
│   ├── binary_positional_encoding.py
│   └── index_positional_encoding.py
├── utils/
│   ├── __init__.py
│   └── math_utils.py      # Helper functions
├── visualizations/
│   ├── __init__.py
│   ├── plot_heatmap.py
│   ├── plot_frequency.py
│   ├── plot_rotation.py
│   └── plot_comparison.py
└── docs/
    ├── SINUSOIDAL.md      # Detailed explanations
    ├── ROTARY.md
    ├── ALIBI.md
    ├── BINARY.md
    ├── INDEX.md
    └── COMPARISON.md      # Comparison guide
```

## Quick Start

### Basic Usage

```python
from encodings import SinusoidalPE, RotaryPE, ALiBiPE
from visualizations import plot_heatmap
import torch

pe = SinusoidalPE(d_model=64, seq_len=128)()
plot_heatmap(pe, "Sinusoidal Positional Encoding")

rope = RotaryPE(d_model=64)
x = torch.randn(128, 64)
positions = torch.arange(128, dtype=torch.float32)
rotated = rope(x, positions)

alibi = ALiBiPE(num_heads=8)
bias = alibi(seq_len=128)
```

### Running All Demonstrations

```bash
uv run python -m positional-encoding.main
```

This will display:
- Individual encoder visualizations
- Frequency analysis
- Comparative heatmaps
- 3D projections

## Configuration

Modify parameters in `config/__init__.py`:

```python
D_MODEL = 64              # Embedding dimension
SEQ_LEN = 128            # Sequence length
NUM_HEADS = 8            # Number of attention heads (ALiBi)
MAX_SEQ_LEN = 2048       # Maximum sequence length
BASE_FREQUENCY = 10000.0 # Base frequency for sinusoidal/rotary
```

## API Reference

### SinusoidalPE

```python
encoder = SinusoidalPE(d_model: int, seq_len: int)
pe = encoder()  # Returns: torch.Tensor of shape (seq_len, d_model)
```

### RotaryPE

```python
encoder = RotaryPE(d_model: int, max_seq_len: int = 2048, base: float = 10000.0)
rotated = encoder(x: torch.Tensor, positions: torch.Tensor = None)
```

### ALiBiPE

```python
encoder = ALiBiPE(num_heads: int, max_seq_len: int = 2048)
bias = encoder(seq_len: int)  # Returns: torch.Tensor of shape (num_heads, seq_len, seq_len)
```

### BinaryPE

```python
encoder = BinaryPE(seq_len: int)
pe = encoder()  # Returns: torch.Tensor of shape (seq_len, num_bits)
```

### IndexPE

```python
encoder = IndexPE(seq_len: int, normalize: bool = False)
pe = encoder()  # Returns: torch.Tensor of shape (seq_len, 1)
```

## Visualization Functions

### Heatmap

```python
from visualizations import plot_heatmap

plot_heatmap(pe, title="My Encoding", save_path="output.png", figsize=(12, 8))
```

### Frequency Analysis

```python
from visualizations import plot_frequency

plot_frequency(pe, dims=[0, 2, 4, 8], title="Frequency Components")
```

### Rotation Visualization

```python
from visualizations import plot_rotation

plot_rotation(d_model=64, num_positions=100)
```

### Comparison

```python
from visualizations import plot_comparison

encodings = {
    "Sinusoidal": sinusoidal_pe,
    "Binary": binary_pe,
}
plot_comparison(encodings)
```

### 3D Visualization

```python
from visualizations import plot_3d_encoding

plot_3d_encoding(pe, title="3D View")
```

## Utility Functions

```python
from utils import (
    compute_frequencies,
    apply_rotary_embedding,
    relative_position_matrix,
    normalize_encoding,
    interpolate_encoding,
)

freqs = compute_frequencies(d_model=64, base=10000.0)

normalized_pe = normalize_encoding(pe, method='l2')

extended_pe = interpolate_encoding(pe, target_len=256)
```

## Understanding the Encodings

Each encoding method has detailed documentation in the `docs/` folder:

- **[SINUSOIDAL.md](docs/SINUSOIDAL.md)**: Mathematical foundation, properties, use cases
- **[ROTARY.md](docs/ROTARY.md)**: Rotation mechanism, advantages, modern applications
- **[ALIBI.md](docs/ALIBI.md)**: Linear bias approach, extrapolation capabilities
- **[BINARY.md](docs/BINARY.md)**: Binary representation, compactness analysis
- **[INDEX.md](docs/INDEX.md)**: Simple indexing, use with learned embeddings
- **[COMPARISON.md](docs/COMPARISON.md)**: Side-by-side comparison and selection guide

## Choosing the Right Encoding

**For general transformer tasks**: Start with Sinusoidal

**For large language models**: Use Rotary (RoPE)

**For length extrapolation**: Choose ALiBi or Rotary

**For memory-constrained scenarios**: Consider Binary

**For learned embeddings**: Use Index as input

See [COMPARISON.md](docs/COMPARISON.md) for detailed guidance.

## Examples

### Example 1: Compare Encodings

```python
import torch
from encodings import SinusoidalPE, BinaryPE, IndexPE
from visualizations import plot_comparison

d_model, seq_len = 64, 128

encodings = {
    "Sinusoidal": SinusoidalPE(d_model, seq_len)(),
    "Binary": BinaryPE(seq_len)(),
    "Index": IndexPE(seq_len, normalize=True)(),
}

plot_comparison(encodings)
```

### Example 2: Apply RoPE to Attention

```python
import torch
from encodings import RotaryPE

batch_size, num_heads, seq_len, head_dim = 8, 12, 128, 64

rope = RotaryPE(d_model=head_dim)

queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
keys = torch.randn(batch_size, num_heads, seq_len, head_dim)

positions = torch.arange(seq_len, dtype=torch.float32)

q_rotated = rope(queries, positions)
k_rotated = rope(keys, positions)

attention_scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1))
```

### Example 3: Use ALiBi in Attention

```python
import torch
import torch.nn.functional as F
from encodings import ALiBiPE

num_heads, seq_len = 8, 128

alibi = ALiBiPE(num_heads=num_heads)
bias = alibi(seq_len=seq_len)

queries = torch.randn(1, num_heads, seq_len, 64)
keys = torch.randn(1, num_heads, seq_len, 64)

scores = torch.matmul(queries, keys.transpose(-2, -1))
scores = scores + bias.unsqueeze(0)

attention_weights = F.softmax(scores / (64 ** 0.5), dim=-1)
```

## Performance Considerations

**Sinusoidal**: Can be precomputed and cached for efficiency

**Rotary**: Computed per forward pass, moderate overhead

**ALiBi**: Very lightweight, just bias addition

**Binary**: Minimal computation and memory

**Index**: Negligible overhead

## Advanced Usage

### Custom Normalization

```python
from utils import normalize_encoding

pe = SinusoidalPE(64, 128)()
pe_l2 = normalize_encoding(pe, method='l2')
pe_minmax = normalize_encoding(pe, method='min_max')
pe_standard = normalize_encoding(pe, method='standard')
```

### Interpolation for Different Lengths

```python
from utils import interpolate_encoding

pe_128 = SinusoidalPE(64, 128)()
pe_256 = interpolate_encoding(pe_128, target_len=256)
```

## References

1. Vaswani et al. (2017) - "Attention is All You Need"
2. Su et al. (2021) - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
3. Press et al. (2021) - "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"

## License

See LICENSE file for details.

## Contributing

This is an educational and research-focused implementation. The code is designed for clarity and understanding rather than production optimization.

## Acknowledgments

Implementations based on original papers and reference implementations from the transformer research community.
