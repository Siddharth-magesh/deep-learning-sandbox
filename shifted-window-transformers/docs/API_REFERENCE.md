# API Reference

Complete API documentation for all modules and classes.

## Configuration

### SwimConfig

```python
from shifted_window_transformers.src.config import SwimConfig

@dataclass
class SwimConfig:
    # Image settings
    image_size: int = 224        # Input image size
    patch_size: int = 4          # Patch size
    in_channels: int = 3         # Input channels
    num_classes: int = 1000      # Output classes
    
    # Model architecture
    embed_dim: int = 96          # Base embedding dimension
    depths: List[int]            # [2, 2, 6, 2] - blocks per stage
    num_heads: List[int]         # [3, 6, 12, 24] - heads per stage
    window_size: int = 7         # Window size for attention
    mlp_ratio: float = 4.0       # MLP expansion ratio
    qkv_bias: bool = True        # Bias in QKV projection
    
    # Regularization
    drop_rate: float = 0.0       # Dropout rate
    attn_drop_rate: float = 0.0  # Attention dropout
    drop_path_rate: float = 0.1  # Stochastic depth rate
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
```

### DataConfig

```python
from shifted_window_transformers.src.config import DataConfig

@dataclass
class DataConfig:
    dataset_name: str = "tiny-imagenet"
    data_dir: str = "./data/tiny-imagenet"
    batch_size: int = 128
    num_workers: int = os.cpu_count()
    pin_memory: bool = True
    mean: tuple = (0.4802, 0.4481, 0.3975)
    std: tuple = (0.2302, 0.2265, 0.2262)
    image_size: int = 224
```

---

## Models

### SwinTransformer

Main model class implementing the full Swin Transformer.

```python
from shifted_window_transformers.src.models import SwinTransformer

class SwinTransformer(nn.Module):
    def __init__(self, config: SwimConfig) -> None
    def forward(self, x: Tensor, labels: Tensor = None) -> Tuple[Tensor, Tensor]
    def forward_features(self, x: Tensor) -> Tensor
```

**Parameters:**
- `config` (SwimConfig): Model configuration

**Forward Arguments:**
- `x` (Tensor): Input images, shape `(B, C, H, W)`
- `labels` (Tensor, optional): Target labels, shape `(B,)`

**Returns:**
- `logits` (Tensor): Class logits, shape `(B, num_classes)`
- `loss` (Tensor): Cross-entropy loss if labels provided, else None

**Example:**
```python
config = SwimConfig(num_classes=200)
model = SwinTransformer(config)

images = torch.randn(4, 3, 224, 224)
logits, _ = model(images)  # (4, 200)

# With labels
labels = torch.randint(0, 200, (4,))
logits, loss = model(images, labels)
```

### Model Factory Functions

```python
from shifted_window_transformers.src.models import swin_tiny, swin_small, swin_base

# Swin-Tiny (~28M params)
model = swin_tiny(num_classes=200, image_size=224)

# Swin-Small (~50M params)
model = swin_small(num_classes=200, image_size=224)

# Swin-Base (~88M params)
model = swin_base(num_classes=200, image_size=224)
```

---

## Modules

### PatchEmbed

Patch embedding layer using 2D convolution.

```python
from shifted_window_transformers.src.modules import PatchEmbed

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int,      # Input image size
        patch_size: int,    # Patch size
        in_channels: int,   # Input channels
        embed_dim: int      # Embedding dimension
    ) -> None
    
    def forward(self, x: Tensor) -> Tensor
```

**Input:** `(B, C, H, W)` → **Output:** `(B, H/P, W/P, embed_dim)`

### WindowAttention

Window-based multi-head self-attention with relative position bias.

```python
from shifted_window_transformers.src.modules import WindowAttention

class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,           # Input dimension
        window_size: int,   # Window size
        num_heads: int,     # Number of attention heads
        qkv_bias: bool,     # Use bias in QKV
        attn_drop: float,   # Attention dropout
        proj_drop: float    # Projection dropout
    ) -> None
    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor
```

**Input:** `(B*num_windows, window_size², C)` → **Output:** same shape

### SwinTransformerBlock

Single Swin Transformer block with W-MSA or SW-MSA.

```python
from shifted_window_transformers.src.modules import SwinTransformerBlock

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,                          # Input dimension
        input_resolution: Tuple[int, int], # (H, W)
        num_heads: int,                    # Attention heads
        window_size: int = 7,              # Window size
        shift_size: int = 0,               # Shift size (0 or window_size//2)
        mlp_ratio: float = 4.0,            # MLP expansion
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ) -> None
    
    def forward(self, x: Tensor) -> Tensor
```

**Input/Output:** `(B, H, W, C)`

### PatchMerging

Downsampling layer that merges 2×2 patches.

```python
from shifted_window_transformers.src.modules import PatchMerging

class PatchMerging(nn.Module):
    def __init__(
        self,
        input_resolution: Tuple[int, int],  # (H, W)
        dim: int,                            # Input dimension
        norm_layer: nn.Module = nn.LayerNorm
    ) -> None
    
    def forward(self, x: Tensor) -> Tensor
```

**Input:** `(B, H, W, C)` → **Output:** `(B, H/2, W/2, 2C)`

### MLP

Feed-forward network with GELU activation.

```python
from shifted_window_transformers.src.modules import MLP

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0
    ) -> None
    
    def forward(self, x: Tensor) -> Tensor
```

### Window Operations

```python
from shifted_window_transformers.src.modules import window_partition, window_reverse

def window_partition(x: Tensor, window_size: int) -> Tensor:
    """
    Partition feature map into windows.
    
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (B*num_windows, window_size, window_size, C)
    """

def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size: int
        H, W: Original feature map size
    Returns:
        x: (B, H, W, C)
    """
```

---

## Utilities

### DropPath

Stochastic depth regularization.

```python
from shifted_window_transformers.src.utils import DropPath, drop_path

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None
    def forward(self, x: Tensor) -> Tensor

# Functional version
def drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor
```

### Weight Initialization

```python
from shifted_window_transformers.src.utils import trunc_normal_, init_weights_swin

def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0
) -> Tensor:
    """Truncated normal initialization."""

def init_weights_swin(model: nn.Module) -> None:
    """Initialize Swin Transformer weights."""
```

### Model Summary

```python
from shifted_window_transformers.src.utils import print_model_summary

def print_model_summary(
    model: nn.Module,
    input_size: Tuple = (1, 3, 224, 224)
) -> None:
    """Print model parameter count and size."""
```

---

## Optimizer & Scheduler

### build_optimizer

```python
from shifted_window_transformers.src.optim import build_optimizer

def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    momentum: float = 0.9,
    betas: Tuple = (0.9, 0.999)
) -> torch.optim.Optimizer
```

### build_scheduler

```python
from shifted_window_transformers.src.optim import build_scheduler

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_warmup",
    total_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-5,
    steps_per_epoch: int = None
) -> torch.optim.lr_scheduler._LRScheduler
```

### WarmupCosineScheduler

```python
from shifted_window_transformers.src.optim import WarmupCosineScheduler

class WarmupCosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-5,
        last_epoch: int = -1
    ) -> None
```

---

## Training & Evaluation

### Trainer

```python
from shifted_window_transformers.src.train import Trainer

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: SwimConfig,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs/swin_transformer",
        max_epochs: int = 100,
        max_training_hours: float = 24.0,
        gradient_clip: float = 1.0,
        mixed_precision: bool = True,
        log_every: int = 100,
        device: str = "cuda"
    ) -> None
    
    def train(self) -> None
    def train_epoch(self) -> Tuple[float, float]
    def validate(self) -> Tuple[float, float]
    def save_checkpoint(self, filename: str) -> None
    def load_checkpoint(self, checkpoint_path: str) -> None
```

### Evaluator

```python
from shifted_window_transformers.src.evaluate import Evaluator

class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        num_classes: int = 200,
        class_names: List[str] = None,
        device: str = "cuda"
    ) -> None
    
    def evaluate(self) -> Tuple[float, float]
    def get_predictions(self, loader: DataLoader = None) -> Tuple[Tensor, Tensor, Tensor]
```

---

## Data

### TinyImageNetDataset

```python
from shifted_window_transformers.src.data import TinyImageNetDataset

class TinyImageNetDataset:
    def __init__(self, transform: transforms.Compose = None) -> None
    def get_dataset_splits(self) -> Tuple[Dataset, Dataset]
```

### Transforms

```python
from shifted_window_transformers.src.data import train_transformation, test_transformation

def train_transformation(
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    image_size: int
) -> transforms.Compose

def test_transformation(
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    image_size: int
) -> transforms.Compose
```

### DataLoader

```python
from shifted_window_transformers.src.data import get_dataloader

def get_dataloader(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool
) -> Tuple[DataLoader, DataLoader]
```
