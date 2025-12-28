# API Reference

## Configuration Classes

### ViTConfig

Vision Transformer model configuration.

**Attributes:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image_size | int | 32 | Input image size |
| patch_size | int | 4 | Size of image patches |
| in_channels | int | 3 | Number of input channels |
| num_classes | int | 10 | Number of output classes |
| embedding_dim | int | 192 | Embedding dimension |
| num_heads | int | 3 | Number of attention heads |
| num_layers | int | 12 | Number of transformer layers |
| mlp_size | int | 768 | MLP hidden dimension |
| attn_dropout | float | 0.1 | Attention dropout rate |
| mlp_dropout | float | 0.1 | MLP dropout rate |
| initializer_range | float | 0.02 | Weight initialization std |

**Example:**
```python
from vision_transformers.src.config import ViTConfig

config = ViTConfig(
    embedding_dim=384,
    num_heads=6,
    num_layers=12
)
```

### TrainingConfig

Training hyperparameters.

**Attributes:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| batch_size | int | 128 | Training batch size |
| learning_rate | float | 3e-4 | Initial learning rate |
| weight_decay | float | 0.01 | L2 regularization |
| beta1 | float | 0.9 | Adam beta1 |
| beta2 | float | 0.999 | Adam beta2 |
| epsilon | float | 1e-8 | Adam epsilon |
| max_epochs | int | 100 | Maximum training epochs |
| max_training_hours | float | 24.0 | Maximum training time in hours |
| warmup_epochs | int | 5 | LR warmup epochs |
| gradient_clip | float | 1.0 | Gradient clipping threshold |
| save_every | int | 5 | Checkpoint save frequency |
| log_every | int | 50 | Logging frequency |
| eval_every | int | 1 | Validation frequency |
| checkpoint_dir | str | "checkpoints" | Checkpoint directory |
| device | str | auto | Device (cuda/cpu) |
| num_workers | int | 4 | DataLoader workers |
| pin_memory | bool | True | Pin memory for GPU |
| mixed_precision | bool | True | Use AMP |
| label_smoothing | float | 0.1 | Label smoothing |

### DataConfig

Dataset configuration.

**Attributes:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dataset_name | str | "cifar10" | Dataset name |
| data_dir | str | "./data" | Data directory |
| batch_size | int | 128 | Batch size |
| num_workers | int | auto | DataLoader workers |
| pin_memory | bool | True | Pin memory for GPU |
| mean | tuple | CIFAR-10 | Normalization mean |
| std | tuple | CIFAR-10 | Normalization std |
| image_size | int | 32 | Image size |

## Model Classes

### VisionTransformer

Main Vision Transformer model.

**Constructor:**
```python
VisionTransformer(config: ViTConfig)
```

**Methods:**

#### forward
```python
forward(
    x: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

**Parameters:**
- x: Input images (B, 3, 32, 32)
- labels: Target labels (B,) [optional]

**Returns:**
- logits: Class predictions (B, 10)
- loss: Cross-entropy loss [if labels provided]

## Module Classes

### PatchEmbedding

Converts images to patch embeddings.

**Constructor:**
```python
PatchEmbedding(
    image_size: int,
    patch_size: int,
    in_channels: int,
    embedding_dim: int
)
```

**forward:**
```python
forward(x: torch.Tensor) -> torch.Tensor
```

Input: (B, C, H, W)
Output: (B, num_patches+1, embedding_dim)

### MultiHeadAttention

Multi-head self-attention mechanism.

**Constructor:**
```python
MultiHeadAttention(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.1
)
```

**forward:**
```python
forward(x: torch.Tensor) -> torch.Tensor
```

Input: (B, N, D)
Output: (B, N, D)

### TransformerEncoder

Single transformer encoder block.

**Constructor:**
```python
TransformerEncoder(
    embedding_dim: int,
    num_heads: int,
    mlp_size: int,
    attn_dropout: float = 0.1,
    mlp_dropout: float = 0.1
)
```

**forward:**
```python
forward(x: torch.Tensor) -> torch.Tensor
```

Input: (B, N, D)
Output: (B, N, D)

### MultiLayerPerceptron

Feed-forward network.

**Constructor:**
```python
MultiLayerPerceptron(
    embedding_dim: int,
    mlp_size: int,
    dropout: float = 0.1
)
```

**forward:**
```python
forward(x: torch.Tensor) -> torch.Tensor
```

Input: (B, N, D)
Output: (B, N, D)

### MultiHeadPerceptronClassifier

Classification head.

**Constructor:**
```python
MultiHeadPerceptronClassifier(
    embedding_dim: int,
    num_classes: int
)
```

**forward:**
```python
forward(x: torch.Tensor) -> torch.Tensor
```

Input: (B, D)
Output: (B, num_classes)

## Training Classes

### Trainer

Handles model training.

**Constructor:**
```python
Trainer(
    model: VisionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    checkpoint_dir: str
)
```

**Methods:**

#### train
```python
train() -> None
```

Runs full training loop.

#### train_epoch
```python
train_epoch() -> Tuple[float, float]
```

Trains for one epoch.

Returns: (loss, accuracy)

#### validate
```python
validate() -> Tuple[float, float]
```

Validates on validation set.

Returns: (loss, accuracy)

#### save_checkpoint
```python
save_checkpoint(filename: str) -> None
```

Saves model checkpoint.

#### load_checkpoint
```python
load_checkpoint(checkpoint_path: str) -> None
```

Loads model from checkpoint.

## Evaluation Classes

### Evaluator

Handles model evaluation.

**Constructor:**
```python
Evaluator(
    model: VisionTransformer,
    test_loader: DataLoader,
    device: str = 'cuda'
)
```

**Methods:**

#### evaluate
```python
evaluate() -> Tuple[float, float]
```

Evaluates model on test set.

Returns: (loss, accuracy)

## Data Functions

### get_dataloaders
```python
get_dataloaders(
    root: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]
```

Creates train, validation, and test dataloaders.

Returns: (train_loader, val_loader, test_loader)

### get_train_transform
```python
get_train_transform(image_size: int = 32) -> transforms.Compose
```

Returns training data transforms.

### get_test_transform
```python
get_test_transform() -> transforms.Compose
```

Returns test/validation transforms.

## Utility Functions

### print_model_summary
```python
print_model_summary(model: nn.Module) -> None
```

Prints model parameter summary.

### count_parameters_by_layer
```python
count_parameters_by_layer(model: nn.Module) -> Dict[str, int]
```

Returns parameter count per layer.

## CLI Commands

### Training

```bash
uv run python -m vision-transformers.src.main train [OPTIONS]
```

**Options:**
- --data_dir: Data directory
- --batch_size: Batch size
- --learning_rate: Learning rate
- --max_epochs: Number of epochs
- --max_training_hours: Maximum training time
- --image_size: Image size
- --patch_size: Patch size
- --num_classes: Number of classes
- --embedding_dim: Embedding dimension
- --num_heads: Number of attention heads
- --num_layers: Number of layers
- --mlp_size: MLP size
- --checkpoint_dir: Checkpoint directory
- --device: Device (cuda/cpu)
- --resume_from: Resume from checkpoint

### Evaluation

```bash
uv run python -m vision-transformers.src.main evaluate [OPTIONS]
```

**Options:**
- --model_path: Path to checkpoint (required)
- --data_dir: Data directory
- --batch_size: Batch size
- --device: Device (cuda/cpu)
