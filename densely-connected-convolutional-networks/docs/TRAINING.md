# Training Guide

## Overview

This guide covers advanced training techniques, best practices, and optimization strategies for DenseNet models.

## Training Pipeline

### 1. Data Preparation

#### CIFAR-10/100

The CIFAR datasets will be automatically downloaded:

```python
from src.data.dataset import build_dataloaders
from src.config.config import Config

cfg = Config()
cfg.data.dataset = "cifar10"  # or "cifar100"
cfg.data.batch_size = 64
cfg.data.num_workers = 4

train_loader, val_loader = build_dataloaders(cfg)
```

**Data Augmentation:**
- Random crop (32×32 with padding 4)
- Random horizontal flip
- Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

#### ImageNet

For ImageNet, download the dataset and organize it:

```
data/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

Update configuration:
```python
cfg.data.dataset = "imagenet"
cfg.data.data_dir = "./data/imagenet"
cfg.data.batch_size = 256  # Adjust based on GPU memory
```

### 2. Model Selection

Choose the appropriate DenseNet variant based on your requirements:

| Model | Parameters | Memory | Accuracy (CIFAR-10) | Training Time |
|-------|-----------|--------|---------------------|---------------|
| DenseNet-121 | ~7M | ~2GB | ~95% | Fast |
| DenseNet-169 | ~14M | ~3GB | ~95.5% | Medium |
| DenseNet-201 | ~20M | ~4GB | ~96% | Slow |
| DenseNet-264 | ~34M | ~6GB | ~96.2% | Very Slow |

**Configuration:**
```python
# DenseNet-121
cfg.model.name = "densenet121"
cfg.model.block_layers = [6, 12, 24, 16]

# DenseNet-169
cfg.model.name = "densenet169"
cfg.model.block_layers = [6, 12, 32, 32]

# DenseNet-201
cfg.model.name = "densenet201"
cfg.model.block_layers = [6, 12, 48, 32]

# DenseNet-264
cfg.model.name = "densenet264"
cfg.model.block_layers = [6, 12, 64, 48]
```

### 3. Hyperparameter Configuration

#### Learning Rate

**CIFAR-10/100:**
- **SGD:** Start with 0.1, decay by 10× at 50% and 75% of training
- **AdamW:** Start with 0.001, use cosine annealing

**ImageNet:**
- **SGD:** 0.1 with batch size 256
- Linear warmup for first 5 epochs
- Decay at epochs 30, 60, 90

**Example:**
```python
cfg.optimizer.name = "sgd"
cfg.optimizer.lr = 0.1
cfg.optimizer.momentum = 0.9
cfg.optimizer.weight_decay = 1e-4

cfg.scheduler.name = "cosine"
cfg.scheduler.t_max = 200
cfg.scheduler.warmup_epochs = 5
```

#### Batch Size

General rule: Larger batch sizes require higher learning rates.

- **Single GPU:** 64-128 (CIFAR), 32-64 (ImageNet)
- **Multi-GPU:** Scale batch size and learning rate linearly
  - If batch size × N, then learning rate × N

```python
# Single GPU
cfg.data.batch_size = 64
cfg.optimizer.lr = 0.1

# 4 GPUs (effective batch size = 256)
cfg.data.batch_size = 64  # Per GPU
cfg.optimizer.lr = 0.4  # Scaled by 4
```

### 4. Advanced Training Techniques

#### Mixed Precision Training

Reduces memory usage and speeds up training:

```python
cfg.training.mixed_precision = True
```

**Benefits:**
- 2-3× faster training
- ~50% less GPU memory
- Minimal accuracy loss

**Note:** Requires GPU with Tensor Cores (Volta, Turing, Ampere, or newer)

#### Gradient Clipping

Prevents exploding gradients:

```python
cfg.training.grad_clip = 1.0  # Clip gradients to max norm of 1.0
```

**When to use:**
- Training becomes unstable
- Loss oscillates or diverges
- Very deep networks

#### Label Smoothing

Improves generalization:

```python
from torch.nn import CrossEntropyLoss

# Instead of standard cross entropy
criterion = CrossEntropyLoss(label_smoothing=0.1)
```

**Benefits:**
- Prevents overconfidence
- Better calibration
- Slight accuracy improvement

### 5. Training Schedule

#### CIFAR-10 Recommended Schedule

```python
cfg.training.epochs = 200
cfg.data.batch_size = 64
cfg.optimizer.name = "sgd"
cfg.optimizer.lr = 0.1
cfg.optimizer.momentum = 0.9
cfg.optimizer.weight_decay = 1e-4
cfg.scheduler.name = "cosine"
cfg.scheduler.t_max = 200
```

**Expected timeline:**
- Epoch 1-50: Rapid improvement
- Epoch 50-100: Gradual improvement
- Epoch 100-200: Fine-tuning
- Final accuracy: 94-96%

#### ImageNet Recommended Schedule

```python
cfg.training.epochs = 90
cfg.data.batch_size = 256  # Total across all GPUs
cfg.optimizer.name = "sgd"
cfg.optimizer.lr = 0.1
cfg.optimizer.momentum = 0.9
cfg.optimizer.weight_decay = 1e-4
cfg.scheduler.name = "step"
cfg.scheduler.step_size = 30
cfg.scheduler.gamma = 0.1
```

**Learning rate schedule:**
- Epoch 1-30: lr = 0.1
- Epoch 31-60: lr = 0.01
- Epoch 61-90: lr = 0.001

### 6. Monitoring Training

#### Metrics to Track

1. **Training Loss:** Should decrease smoothly
2. **Validation Loss:** Should decrease; if it increases while training loss decreases, you're overfitting
3. **Training Accuracy:** Monitor to ensure model is learning
4. **Validation Accuracy:** Primary metric for model selection
5. **Learning Rate:** Verify scheduler is working correctly

#### Using TensorBoard (Optional)

Add TensorBoard logging:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=cfg.runtime.output_dir)

# In training loop
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
```

Launch TensorBoard:
```bash
tensorboard --logdir=./outputs
```

### 7. Checkpointing Strategy

#### Automatic Checkpointing

The training script automatically saves:
- **best.pth:** Model with best validation accuracy
- **checkpoint_epoch_N.pth:** Regular checkpoints (configurable)

#### Manual Checkpointing

```python
import torch
import os

def save_checkpoint(model, optimizer, epoch, best_acc, cfg):
    os.makedirs(cfg.runtime.output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_acc': best_acc,
        'config': cfg,
    }
    
    path = os.path.join(cfg.runtime.output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")
```

#### Resume Training

```bash
uv run python src/main.py \
    --mode train \
    --model densenet121 \
    --dataset cifar10 \
    --checkpoint ./outputs/checkpoint_epoch_100.pth
```

Or in code:
```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
start_epoch = checkpoint['epoch'] + 1
```

## Optimization Strategies

### 1. Data Loading Optimization

```python
cfg.data.num_workers = 4  # Use multiple workers
cfg.data.pin_memory = True  # Pin memory for faster GPU transfer
```

**Guidelines:**
- num_workers = 4 is a good default
- Too many workers can slow down training due to overhead
- pin_memory=True helps on systems with GPU

### 2. Memory Optimization

If you encounter OOM errors:

1. **Reduce batch size:**
   ```python
   cfg.data.batch_size = 32  # Instead of 64
   ```

2. **Use gradient accumulation:**
   ```python
   accumulation_steps = 4
   optimizer.zero_grad()
   
   for i, (images, targets) in enumerate(dataloader):
       outputs = model(images)
       loss = criterion(outputs, targets) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Use mixed precision:**
   ```python
   cfg.training.mixed_precision = True
   ```

### 3. Speed Optimization

**Enable cuDNN autotuner:**
```python
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
```

**Use DataParallel for multi-GPU:**
```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

**Optimize data augmentation:**
- Use GPU-accelerated transforms when possible
- Reduce num_workers if CPU is bottleneck

## Common Issues and Solutions

### Issue 1: Loss is NaN

**Causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability

**Solutions:**
```python
# Reduce learning rate
cfg.optimizer.lr = 0.01  # Instead of 0.1

# Enable gradient clipping
cfg.training.grad_clip = 1.0

# Use mixed precision (helps with numerical stability)
cfg.training.mixed_precision = True
```

### Issue 2: Overfitting

**Symptoms:**
- Validation loss increases while training loss decreases
- Large gap between train and val accuracy

**Solutions:**
```python
# Increase weight decay
cfg.optimizer.weight_decay = 1e-3  # Instead of 1e-4

# Add dropout
cfg.model.dropout = 0.2

# Use more data augmentation
cfg.data.random_crop = True
cfg.data.random_flip = True

# Train for fewer epochs or use early stopping
```

### Issue 3: Underfitting

**Symptoms:**
- Both training and validation accuracy are low
- Loss plateaus early

**Solutions:**
```python
# Increase model capacity
cfg.model.name = "densenet169"  # Instead of densenet121

# Train for more epochs
cfg.training.epochs = 300

# Increase learning rate
cfg.optimizer.lr = 0.2

# Reduce regularization
cfg.optimizer.weight_decay = 1e-5
```

### Issue 4: Slow Convergence

**Solutions:**
```python
# Use AdamW instead of SGD
cfg.optimizer.name = "adamw"
cfg.optimizer.lr = 0.001

# Increase batch size
cfg.data.batch_size = 128

# Use learning rate warmup
cfg.scheduler.warmup_epochs = 5
```

## Best Practices

1. **Start Simple:** Begin with DenseNet-121 and default hyperparameters
2. **Monitor Early:** Check first few epochs to catch issues early
3. **Save Everything:** Save checkpoints, config, and logs
4. **Validate Frequently:** Validate every epoch to track progress
5. **Use Version Control:** Track configuration changes
6. **Document Experiments:** Keep notes on what works and what doesn't
7. **Reproducibility:** Set random seeds and save exact configuration
8. **Gradual Changes:** Change one hyperparameter at a time

## Hyperparameter Tuning

### Grid Search Example

```python
learning_rates = [0.01, 0.05, 0.1]
weight_decays = [1e-4, 1e-3, 1e-2]

for lr in learning_rates:
    for wd in weight_decays:
        cfg = Config()
        cfg.optimizer.lr = lr
        cfg.optimizer.weight_decay = wd
        cfg.runtime.experiment_name = f"lr{lr}_wd{wd}"
        
        model = DenseNet(cfg.model)
        train_loader, val_loader = build_dataloaders(cfg)
        train(model, train_loader, val_loader, cfg)
```

### Random Search

Better than grid search for high-dimensional spaces:

```python
import random

for i in range(20):  # 20 random trials
    cfg = Config()
    cfg.optimizer.lr = 10 ** random.uniform(-4, -1)  # 1e-4 to 1e-1
    cfg.optimizer.weight_decay = 10 ** random.uniform(-5, -2)  # 1e-5 to 1e-2
    cfg.model.dropout = random.uniform(0, 0.3)
    cfg.runtime.experiment_name = f"trial_{i}"
    
    # Train and evaluate
    ...
```

## Production Checklist

Before deploying your model:

- [ ] Trained for sufficient epochs
- [ ] Validated on separate test set
- [ ] Documented hyperparameters
- [ ] Saved best checkpoint
- [ ] Tested inference speed
- [ ] Verified model size
- [ ] Checked for overfitting
- [ ] Evaluated on diverse data
- [ ] Documented known limitations

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture explanation
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- Original DenseNet paper: https://arxiv.org/abs/1608.06993
