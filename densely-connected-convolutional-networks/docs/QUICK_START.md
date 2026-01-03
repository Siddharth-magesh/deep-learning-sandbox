# Quick Start Guide

## Installation

### Using UV (Recommended)

```bash
# Install uv if you haven't already
pip install uv

# Navigate to project directory
cd densely-connected-convolutional-networks

# Install dependencies
uv pip install -e .
```

### Using pip

```bash
pip install torch torchvision numpy pillow
```

## Basic Usage

### 1. Training on CIFAR-10

The simplest way to get started is training on CIFAR-10:

```bash
# Using uv
uv run python src/main.py --mode train --model densenet121 --dataset cifar10 --epochs 200

# Using standard Python
python src/main.py --mode train --model densenet121 --dataset cifar10 --epochs 200
```

**Expected output:**
```
Building densenet121 model...
Total parameters: 6,964,106
Trainable parameters: 6,964,106
Building data loaders...
Training samples: 50000
Validation samples: 10000
Starting training...
Epoch [1] Step [0/1562] Loss: 2.3026 Acc: 0.0938
...
```

### 2. Quick Training Example (10 epochs)

For a quick test:

```bash
uv run python src/main.py --mode train --model densenet121 --dataset cifar10 --epochs 10 --batch-size 64
```

### 3. Evaluation

After training, evaluate your model:

```bash
uv run python src/main.py --mode eval --model densenet121 --dataset cifar10 --checkpoint ./outputs/best.pth
```

## Command Line Arguments

### Essential Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--mode` | train, eval | train | Training or evaluation mode |
| `--model` | densenet121, densenet169, densenet201, densenet264 | densenet201 | Model variant |
| `--dataset` | cifar10, cifar100, imagenet | cifar10 | Dataset to use |
| `--epochs` | int | 200 | Number of training epochs |
| `--batch-size` | int | 32 | Batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--checkpoint` | path | None | Path to checkpoint file |

### Example Commands

**Train DenseNet-121 on CIFAR-10:**
```bash
uv run python src/main.py \
    --mode train \
    --model densenet121 \
    --dataset cifar10 \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.1
```

**Train DenseNet-169 on CIFAR-100:**
```bash
uv run python src/main.py \
    --mode train \
    --model densenet169 \
    --dataset cifar100 \
    --epochs 300 \
    --batch-size 64
```

**Evaluate a trained model:**
```bash
uv run python src/main.py \
    --mode eval \
    --model densenet121 \
    --dataset cifar10 \
    --checkpoint ./outputs/best.pth
```

## Python API Usage

### Basic Training Script

```python
from src.config.config import Config
from src.models.densenet import DenseNet
from src.data.dataset import build_dataloaders
from src.train import train

# Create configuration
cfg = Config()
cfg.model.num_classes = 10
cfg.data.dataset = "cifar10"
cfg.training.epochs = 100

# Build model
model = DenseNet(cfg.model)

# Build data loaders
train_loader, val_loader = build_dataloaders(cfg)

# Train
train(model, train_loader, val_loader, cfg)
```

### Custom Model Configuration

```python
from src.config.config import Config
from src.models.densenet import DenseNet

# Create custom configuration
cfg = Config()

# Customize model
cfg.model.name = "custom_densenet"
cfg.model.num_classes = 100
cfg.model.growth_rate = 32
cfg.model.block_layers = [6, 12, 24, 16]  # DenseNet-121 architecture
cfg.model.compression_factor = 0.5
cfg.model.dropout = 0.2

# Build model
model = DenseNet(cfg.model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Inference on Single Image

```python
import torch
from PIL import Image
from torchvision import transforms
from src.config.config import Config
from src.models.densenet import DenseNet
from src.evaluate import evaluate_single_image

# Load model
cfg = Config()
model = DenseNet(cfg.model)
checkpoint = torch.load("outputs/best.pth")
model.load_state_dict(checkpoint["model_state"])

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

image = Image.open("path/to/image.jpg")
image_tensor = transform(image)

# Predict
result = evaluate_single_image(model, image_tensor, cfg)
print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## Testing Components

Run the test script to verify all components:

```bash
uv run python test_components.py
```

Expected output:
```
============================================================
Running DenseNet Component Tests
============================================================
...
✓ ALL TESTS PASSED!
============================================================
```

## Configuration Guide

### Model Configuration

Edit `src/config/model_config.py`:

```python
@dataclass
class DenseNetConfig:
    name: str = "densenet201"
    num_classes: int = 1000
    growth_rate: int = 32  # k in the paper
    block_layers: List[int] = field(default_factory=lambda: [6, 12, 48, 32])
    bn_size: int = 4  # Bottleneck multiplier
    compression_factor: float = 0.5  # θ in transition layers
    dropout: float = 0.0
    in_channels: int = 3
    input_size: int = 224
```

### Training Configuration

Edit `src/config/train_config.py`:

```python
@dataclass
class TrainingConfig:
    epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    grad_clip: Optional[float] = None
    log_interval: int = 50
    save_interval: int = 10
```

### Optimizer Configuration

Edit `src/config/optim_config.py`:

```python
@dataclass
class OptimizerConfig:
    name: str = "adamw"  # Options: sgd, adam, adamw
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4

@dataclass
class SchedulerConfig:
    name: Optional[str] = "cosine"  # Options: step, cosine, none
    t_max: int = 200  # For cosine annealing
    step_size: int = 30  # For step LR
    gamma: float = 0.1
```

## Tips for Best Results

### CIFAR-10/100
- **Batch size:** 64-128
- **Learning rate:** Start with 0.1 for SGD, 0.001 for AdamW
- **Epochs:** 200-300
- **Data augmentation:** Random crop + horizontal flip
- **Model:** DenseNet-121 or DenseNet-169

### ImageNet
- **Batch size:** 256 (distributed across GPUs)
- **Learning rate:** 0.1 for SGD with momentum
- **Epochs:** 90-120
- **Scheduler:** Cosine annealing or step decay at epochs 30, 60
- **Model:** DenseNet-201

### General Tips
1. **Use mixed precision** for faster training on modern GPUs
2. **Monitor validation loss** to prevent overfitting
3. **Save checkpoints regularly** in case of interruptions
4. **Use weight decay** (1e-4) for better generalization
5. **Start with smaller models** (DenseNet-121) for faster iteration

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Use gradient accumulation
- Reduce model size (use DenseNet-121 instead of DenseNet-201)

### Slow Training
- Enable mixed precision: `cfg.training.mixed_precision = True`
- Increase batch size if GPU memory allows
- Use multiple workers: `cfg.data.num_workers = 4`

### Poor Accuracy
- Train for more epochs
- Adjust learning rate (try 0.1 for SGD)
- Enable data augmentation
- Increase model capacity (use larger DenseNet variant)

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture explanation
- See [TRAINING.md](TRAINING.md) for advanced training techniques
- Check [API_REFERENCE.md](API_REFERENCE.md) for complete API documentation
