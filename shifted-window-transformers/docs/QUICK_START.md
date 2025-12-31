# Quick Start Guide

Get started with Swin Transformer in minutes.

## Installation

```bash
cd deep-learning-sandbox
uv sync
```

## Training

### Basic Training

Train Swin-Tiny on Tiny-ImageNet:

```bash
uv run python -m shifted-window-transformers.src.main train
```

### Custom Configuration

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

### Resume Training

```bash
uv run python -m shifted-window-transformers.src.main train \
    --resume_from shifted-window-transformers/checkpoints/checkpoint_epoch_50.pth
```

## Evaluation

Evaluate a trained model:

```bash
uv run python -m shifted-window-transformers.src.main evaluate \
    --model_path shifted-window-transformers/checkpoints/best_model.pth
```

## Using the Model in Code

### Basic Usage

```python
import torch
from shifted_window_transformers.src.models import SwinTransformer
from shifted_window_transformers.src.config import SwimConfig

# Create model
config = SwimConfig(
    image_size=224,
    num_classes=200,  # Tiny-ImageNet
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24]
)
model = SwinTransformer(config)

# Forward pass
images = torch.randn(4, 3, 224, 224)
logits, loss = model(images)  # loss is None without labels
print(f"Output shape: {logits.shape}")  # (4, 200)

# With labels (for training)
labels = torch.randint(0, 200, (4,))
logits, loss = model(images, labels)
print(f"Loss: {loss.item()}")
```

### Using Pre-built Variants

```python
from shifted_window_transformers.src.models import swin_tiny, swin_small, swin_base

# Swin-Tiny (~28M params)
model = swin_tiny(num_classes=200)

# Swin-Small (~50M params)
model = swin_small(num_classes=200)

# Swin-Base (~88M params)
model = swin_base(num_classes=200)
```

### Loading a Checkpoint

```python
import torch
from shifted_window_transformers.src.models import SwinTransformer

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pth", map_location="cpu")
config = checkpoint["config"]

# Create model and load weights
model = SwinTransformer(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    images = torch.randn(1, 3, 224, 224)
    logits, _ = model(images)
    predictions = torch.argmax(logits, dim=1)
```

### Feature Extraction

```python
# Get features without classification head
features = model.forward_features(images)
print(f"Feature shape: {features.shape}")  # (batch_size, 768)
```

## Monitoring Training

### TensorBoard

Training logs are saved to `runs/swin_transformer/`:

```bash
tensorboard --logdir runs/swin_transformer
```

Open http://localhost:6006 to view:
- Training/validation loss
- Training/validation accuracy
- Learning rate schedule

## Project Structure

```
shifted-window-transformers/
├── src/
│   ├── config/          # Configuration classes
│   ├── data/            # Dataset and transforms
│   ├── models/          # Swin Transformer model
│   ├── modules/         # Building blocks
│   ├── optim/           # Optimizer and scheduler
│   ├── utils/           # Utilities
│   ├── main.py          # CLI entry point
│   ├── train.py         # Training loop
│   └── evaluate.py      # Evaluation
├── checkpoints/         # Saved models
└── docs/                # Documentation
```

## Common Issues

### Out of Memory

Reduce batch size:
```bash
--batch_size 32
```

Or use gradient accumulation (modify train.py).

### Slow Training

Enable mixed precision (default):
```bash
--mixed_precision
```

### NaN Loss

- Reduce learning rate: `--learning_rate 1e-4`
- Check gradient clipping: `--gradient_clip 1.0`

## Next Steps

- [Training Guide](TRAINING.md) - Detailed training configuration
- [Architecture](ARCHITECTURE.md) - Understanding the model
- [API Reference](API_REFERENCE.md) - Complete API documentation
