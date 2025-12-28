# Vision Transformer (ViT) Implementation

A complete implementation of Vision Transformer from scratch for image classification on CIFAR-10.

## Features

- Full Vision Transformer architecture with patch embedding
- Multi-head self-attention mechanism
- Training pipeline with mixed precision
- TensorBoard integration for monitoring
- Model checkpointing and evaluation
- Production-ready code structure

## Project Structure

```
vision-transformers/
├── src/
│   ├── config/
│   │   ├── model_config.py
│   │   ├── train_config.py
│   │   ├── data_config.py
│   │   └── __init__.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── datamodule.py
│   │   ├── transform.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── vit.py
│   │   └── __init__.py
│   ├── modules/
│   │   ├── patch_embedding.py
│   │   ├── attention.py
│   │   ├── multi_layer_perceptron.py
│   │   ├── transformer_encoder.py
│   │   ├── classifier.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── model_summary.py
│   │   └── __init__.py
│   ├── main.py
│   ├── train.py
│   ├── evaluate.py
│   └── __init__.py
├── checkpoints/
├── data/
├── docs/
└── README.md
```

## Installation

Install dependencies using uv:

```bash
cd d:\\ai_research_learning
uv sync
```

## Quick Start

### Training

Train the Vision Transformer on CIFAR-10:

```bash
uv run python -m vision-transformers.src.main train --max_epochs 100
```

### Evaluation

Evaluate a trained model:

```bash
uv run python -m vision-transformers.src.main evaluate --model_path vision-transformers/checkpoints/best_model.pth
```

## Model Architecture

### Configuration

- Image Size: 32x32 (CIFAR-10)
- Patch Size: 4x4
- Number of Patches: 64
- Embedding Dimension: 192
- Number of Heads: 3
- Number of Layers: 12
- MLP Size: 768
- Number of Classes: 10
- Parameters: ~2.7M

### Components

1. **Patch Embedding**: Converts image into sequence of patch embeddings
2. **Positional Encoding**: Learnable position embeddings
3. **Transformer Encoder**: Multi-head attention + MLP blocks
4. **Classification Head**: Layer normalization + linear layer

## Training Configuration

### Optimizer

- AdamW optimizer
- Learning Rate: 3e-4
- Weight Decay: 0.01
- Betas: (0.9, 0.999)

### Learning Rate Schedule

- Warmup: 5 epochs
- Cosine decay after warmup

### Regularization

- Dropout: 0.1 (attention and MLP)
- Label Smoothing: 0.1
- Gradient Clipping: 1.0

### Training Settings

- Batch Size: 128
- Max Epochs: 100
- Max Training Hours: 24
- Mixed Precision: FP16
- Device: CUDA (if available)

## Dataset

CIFAR-10:
- Training: 50,000 images
- Validation: 10,000 images
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

Data Augmentation:
- Random Crop (32x32 with padding=4)
- Random Horizontal Flip
- Normalization (mean, std from CIFAR-10)

## Commands

### Training with Custom Parameters

```bash
uv run python -m vision-transformers.src.main train \\
    --batch_size 256 \\
    --learning_rate 5e-4 \\
    --max_epochs 200 \\
    --max_training_hours 48 \\
    --embedding_dim 384 \\
    --num_layers 12 \\
    --num_heads 6
```

### Resume Training

```bash
uv run python -m vision-transformers.src.main train \\
    --resume_from vision-transformers/checkpoints/checkpoint_epoch_50.pth \\
    --max_epochs 200
```

### TensorBoard Monitoring

```bash
tensorboard --logdir=runs/vision_transformer
```

Access at: http://localhost:6006

## Expected Performance

After 100 epochs:
- Training Accuracy: 95-98%
- Validation Accuracy: 85-90%

Training time (NVIDIA RTX 3050):
- ~6-8 hours for 100 epochs

## Advanced Usage

### Custom Model Configuration

Create a custom config in your script:

```python
from vision_transformers.src.config import ViTConfig

config = ViTConfig(
    image_size=32,
    patch_size=4,
    embedding_dim=384,
    num_heads=6,
    num_layers=12,
    mlp_size=1536,
    num_classes=10
)
```

### Programmatic Training

```python
from vision_transformers.src.models import VisionTransformer
from vision_transformers.src.config import ViTConfig, TrainingConfig
from vision_transformers.src.data import get_dataloaders
from vision_transformers.src.train import Trainer

model_config = ViTConfig()
train_config = TrainingConfig()

train_loader, val_loader, test_loader = get_dataloaders()

model = VisionTransformer(model_config)
trainer = Trainer(model, train_loader, val_loader, train_config, 'checkpoints')
trainer.train()
```

## Documentation

- [Architecture Details](docs/ARCHITECTURE.md)
- [Training Guide](docs/TRAINING.md)
- [API Reference](docs/API_REFERENCE.md)

## License

MIT License

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
