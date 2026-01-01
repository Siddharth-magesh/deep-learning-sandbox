# Relative Position Bias

A comprehensive implementation of relative position bias mechanisms for attention-based neural networks, particularly useful in vision transformers.

## Overview

This project implements both 1D and 2D relative position bias, demonstrating how to encode positional information without absolute position embeddings.

## Features

- 1D and 2D relative position bias implementations
- Multi-head attention with relative position bias
- Vision Transformer with relative position bias support
- Training and evaluation utilities
- Visualization tools for attention and bias patterns

## Installation

```bash
uv pip install -r requirements.txt
```

## Quick Start

### Run Demo

```bash
python demo.py
```

This will generate visualizations of 1D and 2D relative position bias patterns.

### Train Model

```bash
python main.py --config configs/rpb_config.yaml
```

## Project Structure

```
relative-position-bias/
├── configs/
│   └── rpb_config.yaml
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   └── transformer.py
├── modules/
│   ├── __init__.py
│   ├── attention.py
│   └── relative_position.py
├── experiments/
│   ├── __init__.py
│   ├── config_utils.py
│   ├── train.py
│   └── visualize.py
├── docs/
├── main.py
├── demo.py
├── requirements.txt
└── README.md
```

## Configuration

Edit `configs/rpb_config.yaml` to customize:

- Model architecture (embedding dimensions, number of heads)
- Relative position bias settings (1D/2D, window size)
- Training parameters
- Visualization options

## Usage Examples

### Using Relative Position Bias

```python
from modules import RelativePositionBias

rpb = RelativePositionBias(
    num_heads=8,
    window_size=(7, 7),
    bias_type="2d",
    init_std=0.02
)

bias = rpb()
```

### Using Multi-Head Attention with RPB

```python
from modules import MultiHeadAttention

attn = MultiHeadAttention(
    embed_dim=96,
    num_heads=4,
    use_relative_position=True,
    rpb_kwargs={'num_heads': 4, 'window_size': (7, 7), 'bias_type': '2d'}
)

output = attn(x)
```

## Documentation

See the `docs/` directory for detailed documentation on:
- Core concepts
- Architecture details
- API reference
- Training guide

## License

MIT License
