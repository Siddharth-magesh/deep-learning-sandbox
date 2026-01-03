# API Reference

## Table of Contents

- [Models](#models)
- [Modules](#modules)
- [Configuration](#configuration)
- [Data Loading](#data-loading)
- [Training](#training)
- [Evaluation](#evaluation)
- [Optimization](#optimization)
- [Utilities](#utilities)

## Models

### DenseNet

```python
class DenseNet(nn.Module)
```

Main DenseNet model implementation.

#### Parameters

- **cfg** (`DenseNetConfig`): Configuration object containing model hyperparameters

#### Attributes

- **stem** (`nn.Sequential`): Initial convolution and pooling layers
- **blocks** (`nn.ModuleList`): List of dense blocks and transition layers
- **final_norm** (`nn.BatchNorm2d`): Final batch normalization
- **classifier** (`nn.Linear`): Classification head

#### Methods

##### `__init__(cfg: DenseNetConfig)`

Initialize the DenseNet model.

**Args:**
- `cfg`: DenseNetConfig object with model specifications

**Example:**
```python
from src.config.model_config import DenseNetConfig
from src.models.densenet import DenseNet

cfg = DenseNetConfig(
    num_classes=10,
    growth_rate=32,
    block_layers=[6, 12, 24, 16]
)
model = DenseNet(cfg)
```

##### `forward(x: torch.Tensor) -> torch.Tensor`

Forward pass through the network.

**Args:**
- `x`: Input tensor of shape `(B, C, H, W)`
  - B: batch size
  - C: channels (typically 3 for RGB)
  - H: height
  - W: width

**Returns:**
- Output tensor of shape `(B, num_classes)` with class logits

**Example:**
```python
import torch
x = torch.randn(4, 3, 224, 224)
output = model(x)  # Shape: (4, 1000)
```

## Modules

### DenseBlock

```python
class DenseBlock(nn.Module)
```

A dense block containing multiple densely connected layers.

#### Parameters

- **num_layers** (`int`): Number of layers in the block
- **in_channels** (`int`): Number of input channels
- **growth_rate** (`int`): Growth rate (k in the paper)
- **bn_size** (`int`): Bottleneck size multiplier
- **dropout** (`float`): Dropout probability (default: 0.0)

#### Attributes

- **layers** (`nn.ModuleList`): List of DenseLayer modules
- **out_channels** (`int`): Number of output channels

#### Methods

##### `forward(x: torch.Tensor) -> torch.Tensor`

**Args:**
- `x`: Input tensor of shape `(B, in_channels, H, W)`

**Returns:**
- Output tensor of shape `(B, out_channels, H, W)`
  - where `out_channels = in_channels + num_layers * growth_rate`

**Example:**
```python
from src.modules.dense_block import DenseBlock

block = DenseBlock(
    num_layers=6,
    in_channels=64,
    growth_rate=32,
    bn_size=4
)
x = torch.randn(4, 64, 56, 56)
out = block(x)  # Shape: (4, 256, 56, 56)
```

### DenseLayer

```python
class DenseLayer(nn.Module)
```

A single dense layer with bottleneck design.

#### Parameters

- **in_channels** (`int`): Number of input channels
- **growth_rate** (`int`): Number of output channels
- **bn_size** (`int`): Bottleneck size multiplier
- **dropout** (`float`): Dropout probability (default: 0.0)

#### Methods

##### `forward(x: torch.Tensor) -> torch.Tensor`

**Args:**
- `x`: Input tensor of shape `(B, in_channels, H, W)`

**Returns:**
- Output tensor of shape `(B, in_channels + growth_rate, H, W)`
  - Concatenation of input and new features

**Example:**
```python
from src.modules.dense_layer import DenseLayer

layer = DenseLayer(
    in_channels=64,
    growth_rate=32,
    bn_size=4
)
x = torch.randn(4, 64, 56, 56)
out = layer(x)  # Shape: (4, 96, 56, 56)
```

### TransitionLayer

```python
class TransitionLayer(nn.Module)
```

Transition layer between dense blocks.

#### Parameters

- **in_channels** (`int`): Number of input channels
- **compression_factor** (`float`): Compression factor θ (default: 0.5)
  - Must be in range (0, 1]

#### Attributes

- **out_channels** (`int`): Number of output channels
  - `out_channels = int(in_channels * compression_factor)`

#### Methods

##### `forward(x: torch.Tensor) -> torch.Tensor`

**Args:**
- `x`: Input tensor of shape `(B, in_channels, H, W)`

**Returns:**
- Output tensor of shape `(B, out_channels, H//2, W//2)`
  - Spatial dimensions reduced by half

**Example:**
```python
from src.modules.transition_layer import TransitionLayer

transition = TransitionLayer(
    in_channels=256,
    compression_factor=0.5
)
x = torch.randn(4, 256, 56, 56)
out = transition(x)  # Shape: (4, 128, 28, 28)
```

## Configuration

### Config

```python
@dataclass
class Config
```

Main configuration class combining all sub-configurations.

#### Attributes

- **model** (`DenseNetConfig`): Model configuration
- **data** (`DataConfig`): Data loading configuration
- **optimizer** (`OptimizerConfig`): Optimizer configuration
- **scheduler** (`SchedulerConfig`): Learning rate scheduler configuration
- **training** (`TrainingConfig`): Training configuration
- **runtime** (`RuntimeConfig`): Runtime configuration

**Example:**
```python
from src.config.config import Config

cfg = Config()
print(cfg.model.growth_rate)  # 32
print(cfg.optimizer.lr)  # 0.001
```

### DenseNetConfig

```python
@dataclass
class DenseNetConfig
```

Model architecture configuration.

#### Attributes

- **name** (`str`): Model name (default: "densenet201")
- **num_classes** (`int`): Number of output classes (default: 1000)
- **growth_rate** (`int`): Growth rate k (default: 32)
- **block_layers** (`List[int]`): Number of layers in each block (default: [6, 12, 48, 32])
- **bn_size** (`int`): Bottleneck multiplier (default: 4)
- **compression_factor** (`float`): Compression in transitions (default: 0.5)
- **dropout** (`float`): Dropout probability (default: 0.0)
- **in_channels** (`int`): Input channels (default: 3)
- **input_size** (`int`): Input image size (default: 224)

### DataConfig

```python
@dataclass
class DataConfig
```

Data loading and augmentation configuration.

#### Attributes

- **dataset** (`str`): Dataset name (default: "imagenet")
  - Options: "cifar10", "cifar100", "imagenet"
- **data_dir** (`str`): Data directory path (default: "./data")
- **batch_size** (`int`): Batch size (default: 32)
- **num_workers** (`int`): Number of data loading workers
- **pin_memory** (`bool`): Pin memory for faster GPU transfer (default: True)
- **random_crop** (`bool`): Apply random crop (default: True)
- **random_flip** (`bool`): Apply random horizontal flip (default: True)
- **normalize** (`bool`): Apply normalization (default: True)

### OptimizerConfig

```python
@dataclass
class OptimizerConfig
```

Optimizer configuration.

#### Attributes

- **name** (`str`): Optimizer name (default: "adamw")
  - Options: "sgd", "adam", "adamw"
- **lr** (`float`): Learning rate (default: 0.001)
- **momentum** (`float`): Momentum for SGD (default: 0.9)
- **weight_decay** (`float`): Weight decay (default: 1e-4)
- **eps** (`float`): Epsilon for Adam/AdamW (default: 1e-8)
- **betas** (`tuple`): Beta parameters for Adam/AdamW (default: (0.9, 0.999))

### SchedulerConfig

```python
@dataclass
class SchedulerConfig
```

Learning rate scheduler configuration.

#### Attributes

- **name** (`Optional[str]`): Scheduler name (default: "cosine")
  - Options: "step", "cosine", None
- **step_size** (`int`): Step size for StepLR (default: 30)
- **gamma** (`float`): Decay factor for StepLR (default: 0.1)
- **t_max** (`int`): Max iterations for CosineAnnealingLR (default: 200)
- **min_lr** (`float`): Minimum learning rate (default: 0.0)
- **step_per** (`str`): Step scheduler per "epoch" or "step" (default: "epoch")
- **warmup_epochs** (`int`): Number of warmup epochs (default: 0)

### TrainingConfig

```python
@dataclass
class TrainingConfig
```

Training process configuration.

#### Attributes

- **epochs** (`int`): Number of epochs (default: 200)
- **device** (`str`): Device to use (default: "cuda" if available else "cpu")
- **mixed_precision** (`bool`): Use mixed precision training (default: True)
- **grad_clip** (`Optional[float]`): Gradient clipping threshold (default: None)
- **log_interval** (`int`): Logging frequency in steps (default: 50)
- **save_interval** (`int`): Checkpoint saving frequency in epochs (default: 10)

### RuntimeConfig

```python
@dataclass
class RuntimeConfig
```

Runtime and experiment configuration.

#### Attributes

- **seed** (`int`): Random seed (default: 42)
- **output_dir** (`str`): Output directory (default: "./outputs")
- **experiment_name** (`str`): Experiment name (default: "densenet201_baseline")

## Data Loading

### build_dataloaders

```python
def build_dataloaders(cfg: Config) -> Tuple[DataLoader, Optional[DataLoader]]
```

Build training and validation data loaders.

**Args:**
- `cfg`: Configuration object

**Returns:**
- Tuple of `(train_loader, val_loader)`

**Example:**
```python
from src.data.dataset import build_dataloaders
from src.config.config import Config

cfg = Config()
cfg.data.dataset = "cifar10"
cfg.data.batch_size = 64

train_loader, val_loader = build_dataloaders(cfg)
```

### build_test_loader

```python
def build_test_loader(cfg: Config) -> DataLoader
```

Build test data loader.

**Args:**
- `cfg`: Configuration object

**Returns:**
- Test data loader

### get_transforms

```python
def get_transforms(cfg: Config, is_train: bool = True) -> transforms.Compose
```

Get data transforms for the specified dataset.

**Args:**
- `cfg`: Configuration object
- `is_train`: Whether to use training transforms

**Returns:**
- Composed torchvision transforms

## Training

### train

```python
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    cfg: Config
) -> None
```

Main training loop.

**Args:**
- `model`: DenseNet model to train
- `train_loader`: Training data loader
- `val_loader`: Validation data loader (optional)
- `cfg`: Configuration object

**Side Effects:**
- Trains the model in-place
- Saves checkpoints to `cfg.runtime.output_dir`
- Prints training progress

**Example:**
```python
from src.train import train

model = DenseNet(cfg.model)
train_loader, val_loader = build_dataloaders(cfg)
train(model, train_loader, val_loader, cfg)
```

### train_one_epoch

```python
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    scaler: Optional[GradScaler],
    epoch: int,
    cfg: Config
) -> Dict[str, float]
```

Train for one epoch.

**Args:**
- `model`: Model to train
- `dataloader`: Training data loader
- `criterion`: Loss function
- `optimizer`: Optimizer
- `scheduler`: Learning rate scheduler (optional)
- `device`: Device to train on
- `scaler`: Gradient scaler for mixed precision (optional)
- `epoch`: Current epoch number
- `cfg`: Configuration object

**Returns:**
- Dictionary with training metrics:
  - `"loss"`: Average training loss
  - `"accuracy"`: Average training accuracy

### validate

```python
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]
```

Validate the model.

**Args:**
- `model`: Model to validate
- `dataloader`: Validation data loader
- `criterion`: Loss function
- `device`: Device to use

**Returns:**
- Dictionary with validation metrics:
  - `"val_loss"`: Average validation loss
  - `"val_accuracy"`: Average validation accuracy

### save_checkpoint

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    cfg: Config,
    filename: str
) -> None
```

Save model checkpoint.

**Args:**
- `model`: Model to save
- `optimizer`: Optimizer state to save
- `epoch`: Current epoch
- `best_acc`: Best validation accuracy achieved
- `cfg`: Configuration object
- `filename`: Checkpoint filename

## Evaluation

### evaluate

```python
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: Config
) -> Dict[str, float]
```

Evaluate model on test dataset.

**Args:**
- `model`: Model to evaluate
- `dataloader`: Test data loader
- `cfg`: Configuration object

**Returns:**
- Dictionary with evaluation metrics:
  - `"test_loss"`: Average test loss
  - `"test_accuracy"`: Average test accuracy
  - `"test_top5_accuracy"`: Top-5 accuracy (if num_classes > 5)

**Example:**
```python
from src.evaluate import evaluate

model = DenseNet(cfg.model)
checkpoint = torch.load("best.pth")
model.load_state_dict(checkpoint["model_state"])

_, test_loader = build_dataloaders(cfg)
metrics = evaluate(model, test_loader, cfg)
print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
```

### evaluate_single_image

```python
@torch.no_grad()
def evaluate_single_image(
    model: nn.Module,
    image: torch.Tensor,
    cfg: Config
) -> Dict[str, any]
```

Evaluate a single image.

**Args:**
- `model`: Model to use
- `image`: Input image tensor of shape `(C, H, W)` or `(1, C, H, W)`
- `cfg`: Configuration object

**Returns:**
- Dictionary with prediction results:
  - `"predicted_class"`: Predicted class index
  - `"confidence"`: Confidence score
  - `"top5_classes"`: Top-5 class indices
  - `"top5_probs"`: Top-5 probabilities

## Optimization

### build_optimizer

```python
def build_optimizer(
    model: nn.Module,
    cfg: OptimizerConfig
) -> torch.optim.Optimizer
```

Build optimizer based on configuration.

**Args:**
- `model`: Model whose parameters to optimize
- `cfg`: Optimizer configuration

**Returns:**
- PyTorch optimizer

**Supported optimizers:**
- SGD
- Adam
- AdamW

### build_scheduler

```python
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: SchedulerConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]
```

Build learning rate scheduler.

**Args:**
- `optimizer`: Optimizer to schedule
- `cfg`: Scheduler configuration

**Returns:**
- Learning rate scheduler or None

**Supported schedulers:**
- StepLR
- CosineAnnealingLR

### build_optimizer_and_scheduler

```python
def build_optimizer_and_scheduler(
    model: nn.Module,
    optim_cfg: OptimizerConfig,
    sched_cfg: SchedulerConfig
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]
```

Build both optimizer and scheduler.

**Args:**
- `model`: Model to optimize
- `optim_cfg`: Optimizer configuration
- `sched_cfg`: Scheduler configuration

**Returns:**
- Tuple of `(optimizer, scheduler)`

## Utilities

### AverageMeter

```python
class AverageMeter
```

Computes and stores the average and current value.

#### Methods

##### `__init__()`

Initialize the meter.

##### `reset()`

Reset all statistics.

##### `update(val: float, n: int = 1)`

Update the meter with a new value.

**Args:**
- `val`: New value to add
- `n`: Number of samples this value represents (default: 1)

**Example:**
```python
from src.utils.meters import AverageMeter

loss_meter = AverageMeter()
acc_meter = AverageMeter()

for batch in dataloader:
    loss = compute_loss(batch)
    acc = compute_accuracy(batch)
    
    loss_meter.update(loss.item(), batch_size)
    acc_meter.update(acc, batch_size)

print(f"Average Loss: {loss_meter.avg:.4f}")
print(f"Average Accuracy: {acc_meter.avg:.4f}")
```

## Command Line Interface

### main.py

Main entry point for training and evaluation.

**Usage:**
```bash
python src/main.py [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | str | train | Mode: train or eval |
| `--config` | str | None | Path to config file |
| `--checkpoint` | str | None | Path to checkpoint |
| `--model` | str | densenet201 | Model variant |
| `--dataset` | str | cifar10 | Dataset name |
| `--epochs` | int | None | Number of epochs |
| `--batch-size` | int | None | Batch size |
| `--lr` | float | None | Learning rate |

**Examples:**

```bash
# Train DenseNet-121 on CIFAR-10
python src/main.py --mode train --model densenet121 --dataset cifar10 --epochs 200

# Evaluate trained model
python src/main.py --mode eval --model densenet121 --dataset cifar10 --checkpoint best.pth
```
