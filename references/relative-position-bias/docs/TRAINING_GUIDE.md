# Training Guide

## Setup

### Installation

Install required dependencies:

```bash
uv pip install -r requirements.txt
```

### Configuration

Edit `configs/rpb_config.yaml` to customize training:

```yaml
project:
  name: "relative_position_bias"
  seed: 42
  device: "cuda"

input:
  batch_size: 32
  sequence_length: 49
  embed_dim: 96

attention:
  num_heads: 4
  head_dim: 24
  dropout: 0.1
  scale: true

relative_position_bias:
  enabled: true
  type: "2d"
  window_size: [7, 7]
  learnable: true
  init_std: 0.02
```

## Training from Scratch

### Basic Training

```bash
python main.py --config configs/rpb_config.yaml
```

### Custom Configuration

Create a new config file:

```yaml
project:
  name: "my_experiment"
  seed: 123
  device: "cuda"

input:
  batch_size: 64
  embed_dim: 192

attention:
  num_heads: 8
  dropout: 0.1

relative_position_bias:
  enabled: true
  type: "2d"
  window_size: [14, 14]
```

Run with custom config:

```bash
python main.py --config configs/my_experiment.yaml
```

## Training Process

### Data Loading

The training script uses `SyntheticImageDataset` for demonstration:

```python
from data import SyntheticImageDataset
from torch.utils.data import DataLoader

train_dataset = SyntheticImageDataset(
    num_samples=1000,
    img_size=224,
    patch_size=16,
    in_channels=3
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### Model Creation

```python
from models import VisionTransformer

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    use_relative_position=True,
    rpb_kwargs={
        'num_heads': 12,
        'window_size': (14, 14),
        'bias_type': '2d',
        'init_std': 0.02
    }
)
```

### Training Loop

```python
from experiments import train_epoch, evaluate, save_checkpoint
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.05
)

num_epochs = 100
best_acc = 0.0

for epoch in range(1, num_epochs + 1):
    train_metrics = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    val_metrics = evaluate(
        model, val_loader, criterion, device
    )
    
    if val_metrics['accuracy'] > best_acc:
        best_acc = val_metrics['accuracy']
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            'checkpoints/best_model.pth'
        )
```

## Hyperparameter Tuning

### Learning Rate

Start with:
- AdamW: 1e-3 to 5e-4
- SGD: 1e-1 to 1e-2

Use learning rate warmup:
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
cosine = CosineAnnealingLR(optimizer, T_max=90)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[10]
)
```

### Weight Decay

Typical values: 0.01 to 0.1

Exclude bias and LayerNorm from weight decay:
```python
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() 
                   if 'bias' not in n and 'norm' not in n],
        'weight_decay': 0.05
    },
    {
        'params': [p for n, p in model.named_parameters() 
                   if 'bias' in n or 'norm' in n],
        'weight_decay': 0.0
    }
]
optimizer = optim.AdamW(param_groups, lr=1e-3)
```

### Dropout

- Start with 0.1
- Increase for smaller datasets
- Decrease for larger datasets
- Apply to attention and MLP

### Relative Position Bias

**Initialization**:
- Standard deviation: 0.01 to 0.02
- Smaller values for stability
- Larger values for faster learning

**Window Size**:
- Smaller windows: fewer parameters, local focus
- Larger windows: more parameters, broader context
- Common: 7×7, 14×14

## Monitoring Training

### Metrics to Track

```python
metrics = {
    'train_loss': train_loss,
    'train_accuracy': train_acc,
    'val_loss': val_loss,
    'val_accuracy': val_acc,
    'learning_rate': optimizer.param_groups[0]['lr']
}
```

### Visualization

Use TensorBoard:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_name')

for epoch in range(num_epochs):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
```

View in browser:
```bash
tensorboard --logdir=runs
```

## Debugging

### Check Bias Values

```python
for name, param in model.named_parameters():
    if 'relative_position_bias' in name:
        print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")
```

### Visualize Bias

```python
from experiments import visualize_relative_position_bias

for block in model.blocks:
    if hasattr(block.attn, 'relative_position_bias'):
        bias = block.attn.relative_position_bias()
        visualize_relative_position_bias(
            bias,
            save_path=f'debug/bias_block_{i}.png'
        )
```

### Gradient Flow

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

## Common Issues

### NaN Loss

**Causes**:
- Learning rate too high
- Gradient explosion
- Numerical instability

**Solutions**:
- Reduce learning rate
- Use gradient clipping
- Check initialization
- Use mixed precision training

```python
from torch.nn.utils import clip_grad_norm_

max_norm = 1.0
clip_grad_norm_(model.parameters(), max_norm)
```

### Poor Convergence

**Causes**:
- Learning rate too low
- Insufficient model capacity
- Poor initialization

**Solutions**:
- Increase learning rate
- Increase model depth/width
- Use warmup schedule
- Check data quality

### Overfitting

**Causes**:
- Model too large for dataset
- Insufficient regularization

**Solutions**:
- Increase dropout
- Add data augmentation
- Reduce model size
- Use early stopping

```python
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    val_loss = evaluate(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        save_checkpoint(...)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break
```

## Advanced Techniques

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model, device_ids=[local_rank])
```

### Checkpointing

Save memory with gradient checkpointing:
```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        if self.training:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

## Performance Optimization

### Batch Size

- Larger batches: better GPU utilization, more stable gradients
- Smaller batches: better generalization, less memory
- Find optimal batch size for your GPU

### Number of Workers

```python
import multiprocessing

num_workers = min(4, multiprocessing.cpu_count())
```

### Pin Memory

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    pin_memory=True
)
```

### Compile Model (PyTorch 2.0+)

```python
model = torch.compile(model)
```
