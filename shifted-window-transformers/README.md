# Swin Transformer Implementation

A complete implementation of the Swin Transformer (Shifted Window Transformer) from scratch for image classification on Tiny ImageNet.

## Overview

Swin Transformer is a hierarchical Vision Transformer that uses shifted windows for efficient self-attention computation. This implementation follows the original paper: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030).

## Features

- Full Swin Transformer architecture with hierarchical stages
- Shifted window multi-head self-attention (W-MSA & SW-MSA)
- Patch merging for downsampling between stages
- Relative position bias for attention
- Mixed precision training support
- TensorBoard integration for monitoring
- Model checkpointing and evaluation
- Configurable model variants (Tiny, Small, Base)

## Project Structure

```
shifted-window-transformers/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── swim_config.py          # Model configuration
│   │   └── data_config.py          # Data configuration
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Tiny ImageNet dataset wrapper
│   │   ├── datamodule.py           # DataLoader utilities
│   │   └── transformation.py       # Data augmentations
│   │
│   ├── modules/                    # Reusable building blocks
│   │   ├── __init__.py
│   │   ├── patch_embed.py          # Patch embedding (Conv2d-based)
│   │   ├── window_ops.py           # window_partition & window_reverse
│   │   ├── attention.py            # WindowAttention (W-MSA & SW-MSA)
│   │   ├── mlp.py                  # Feed-forward network
│   │   ├── swin_block.py           # SwinTransformerBlock
│   │   └── patch_merge.py          # PatchMerging
│   │
│   ├── models/                     # Full model architectures
│   │   ├── __init__.py
│   │   ├── swin_stage.py           # Hierarchical stage (deprecated)
│   │   └── swin_transformer.py     # Full Swin Transformer model
│   │
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── optimizer.py            # AdamW optimizer with weight decay
│   │   └── scheduler.py            # Cosine scheduler with warmup
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── drop_path.py            # Stochastic depth
│   │   ├── weight_init.py          # Weight initialization
│   │   └── model_summary.py        # Model summary utilities
│   │
│   ├── __init__.py
│   ├── main.py                     # CLI entry point
│   ├── train.py                    # Training loop
│   └── evaluate.py                 # Evaluation
│
├── checkpoints/                    # Saved model checkpoints
├── docs/
│   └── structure.md                # Project structure reference
└── README.md
```

## Model Architecture

### Swin-Tiny Configuration (Default)

- Image Size: 224x224
- Patch Size: 4x4
- Embedding Dimension: 96
- Depths: [2, 2, 6, 2] (per stage)
- Attention Heads: [3, 6, 12, 24] (per stage)
- Window Size: 7x7
- MLP Ratio: 4.0
- Parameters: ~28M

### Key Components

1. **Patch Embedding**: Splits image into non-overlapping patches using Conv2d
2. **Swin Transformer Block**: Alternates between W-MSA and SW-MSA
3. **Window Attention**: Efficient self-attention within local windows
4. **Shifted Windows**: Enables cross-window connections
5. **Patch Merging**: Downsamples feature maps between stages
6. **Relative Position Bias**: Learnable position encoding for attention

## Installation

```bash
cd deep-learning-sandbox
uv sync
```

## Usage

### Training

Train Swin-Tiny on Tiny ImageNet:

```bash
uv run python -m shifted-window-transformers.src.main train --max_epochs 100
```

With custom configuration:

```bash
uv run python -m shifted-window-transformers.src.main train \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --max_epochs 100 \
    --embed_dim 96 \
    --depths 2 2 6 2 \
    --num_heads 3 6 12 24 \
    --window_size 7
```

### Evaluation

Evaluate a trained model:

```bash
uv run python -m shifted-window-transformers.src.main evaluate \
    --model_path shifted-window-transformers/checkpoints/best_model.pth
```

### Resume Training

Resume from a checkpoint:

```bash
uv run python -m shifted-window-transformers.src.main train \
    --resume_from shifted-window-transformers/checkpoints/checkpoint_epoch_50.pth
```

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 64 | Training batch size |
| learning_rate | 3e-4 | Initial learning rate |
| weight_decay | 0.05 | AdamW weight decay |
| max_epochs | 100 | Maximum training epochs |
| gradient_clip | 1.0 | Gradient clipping value |
| mixed_precision | True | Use AMP for training |

## Model Variants

```python
from shifted_window_transformers.src.models import swin_tiny, swin_small, swin_base

# Swin-Tiny: ~28M params
model = swin_tiny(num_classes=200)

# Swin-Small: ~50M params
model = swin_small(num_classes=200)

# Swin-Base: ~88M params
model = swin_base(num_classes=200)
```

## Dataset

This implementation uses **Tiny ImageNet** from Hugging Face:
- 200 classes
- 100,000 training images
- 10,000 validation images
- 64x64 original size (resized to 224x224)

## TensorBoard

Monitor training progress:

```bash
tensorboard --logdir runs/swin_transformer
```

## References

- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [Official Implementation](https://github.com/microsoft/Swin-Transformer)
- [Tiny ImageNet Dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet)

## License

MIT License
