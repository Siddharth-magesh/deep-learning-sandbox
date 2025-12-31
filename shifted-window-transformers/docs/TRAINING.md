# Training Guide

Comprehensive guide for training Swin Transformer models.

## Training Pipeline Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Dataset   │ ──▶ │   Model     │ ──▶ │   Trainer   │
│ TinyImageNet│     │    Swin     │     │   Loop      │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Transforms  │     │  Optimizer  │     │ Checkpoints │
│ Augmentation│     │   AdamW     │     │ TensorBoard │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Command Line Arguments

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 64 | Training batch size |
| `--learning_rate` | 3e-4 | Base learning rate |
| `--weight_decay` | 0.05 | AdamW weight decay |
| `--max_epochs` | 100 | Maximum training epochs |
| `--max_training_hours` | 24.0 | Time limit in hours |
| `--gradient_clip` | 1.0 | Gradient clipping norm |
| `--mixed_precision` | True | Use AMP (FP16) |

### Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_size` | 224 | Input image resolution |
| `--num_classes` | 200 | Number of output classes |
| `--embed_dim` | 96 | Base embedding dimension |
| `--depths` | [2,2,6,2] | Blocks per stage |
| `--num_heads` | [3,6,12,24] | Attention heads per stage |
| `--window_size` | 7 | Window size for attention |

### Path Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | checkpoints | Save directory |
| `--log_dir` | runs/swin_transformer | TensorBoard logs |
| `--resume_from` | None | Resume checkpoint path |

## Training Configurations

### Swin-Tiny (Default)

```bash
uv run python -m shifted-window-transformers.src.main train \
    --embed_dim 96 \
    --depths 2 2 6 2 \
    --num_heads 3 6 12 24 \
    --batch_size 64 \
    --learning_rate 3e-4
```

**Specs:** ~28M parameters, ~4.5 GFLOPs

### Swin-Small

```bash
uv run python -m shifted-window-transformers.src.main train \
    --embed_dim 96 \
    --depths 2 2 18 2 \
    --num_heads 3 6 12 24 \
    --batch_size 32 \
    --learning_rate 2e-4
```

**Specs:** ~50M parameters, ~8.7 GFLOPs

### Swin-Base

```bash
uv run python -m shifted-window-transformers.src.main train \
    --embed_dim 128 \
    --depths 2 2 18 2 \
    --num_heads 4 8 16 32 \
    --batch_size 16 \
    --learning_rate 1e-4
```

**Specs:** ~88M parameters, ~15.4 GFLOPs

## Optimizer Configuration

### AdamW (Default)

```python
optimizer = AdamW(
    params,
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05
)
```

**Weight Decay Groups:**
- With decay: Linear weights, Conv weights
- Without decay: Biases, LayerNorm parameters

### Learning Rate Schedule

**Warmup + Cosine Annealing:**

```
LR
│
│    ╱────────────────╲
│   ╱                  ╲
│  ╱                    ╲
│ ╱                      ╲
│╱                        ╲____
└────────────────────────────── Epochs
  Warmup     Cosine Decay
  (5 ep)     (95 epochs)
```

```python
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=5,
    total_epochs=100,
    warmup_lr=1e-6,
    min_lr=1e-5
)
```

## Data Augmentation

### Training Transforms

```python
transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262]
    )
])
```

### Advanced Augmentation (Recommended)

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25)
])
```

## Regularization Techniques

### 1. Stochastic Depth (Drop Path)

Randomly drops entire residual branches during training.

```python
drop_path_rate = 0.1  # Linearly increases from 0 to 0.1 across layers
```

### 2. Label Smoothing

```python
loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

### 3. Weight Decay

```python
weight_decay = 0.05  # Applied only to weights, not biases
```

### 4. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Mixed Precision Training

Automatic Mixed Precision (AMP) reduces memory usage and speeds up training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits, loss = model(images, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- ~2x faster training
- ~50% less GPU memory
- Minimal accuracy impact

## Checkpointing

### Checkpoint Contents

```python
checkpoint = {
    "epoch": current_epoch,
    "global_step": step_count,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_acc": best_accuracy,
    "config": model_config,
    "scaler_state_dict": scaler.state_dict()  # If using AMP
}
```

### Checkpoint Schedule

- `best_model.pth` - Saved when validation accuracy improves
- `checkpoint_epoch_N.pth` - Saved every 10 epochs
- `checkpoint_time_limit.pth` - Saved when time limit reached

## Training Tips

### 1. Learning Rate Selection

Start with `3e-4` for Swin-Tiny and reduce for larger models:
- Swin-Tiny: `3e-4`
- Swin-Small: `2e-4`
- Swin-Base: `1e-4`

### 2. Batch Size vs Learning Rate

Scale learning rate linearly with batch size:
```
lr = base_lr × (batch_size / 64)
```

### 3. Training Duration

Typical convergence times on Tiny-ImageNet:
- Swin-Tiny: ~50-100 epochs
- Swin-Small: ~100-150 epochs
- Swin-Base: ~150-200 epochs

### 4. GPU Memory Optimization

If running out of memory:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use smaller image size (e.g., 192×192)
4. Reduce model size

### 5. Debugging NaN/Inf

```python
# Add to training loop
torch.autograd.set_detect_anomaly(True)

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
```

## Monitoring with TensorBoard

### Launch TensorBoard

```bash
tensorboard --logdir runs/swin_transformer --port 6006
```

### Logged Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per step |
| `train/accuracy` | Training accuracy per step |
| `train/learning_rate` | Current learning rate |
| `val/loss` | Validation loss per epoch |
| `val/accuracy` | Validation accuracy per epoch |

## Expected Results

### Tiny-ImageNet (200 classes)

| Model | Top-1 Acc | Epochs | Time (V100) |
|-------|-----------|--------|-------------|
| Swin-Tiny | ~65-70% | 100 | ~8h |
| Swin-Small | ~68-73% | 100 | ~16h |
| Swin-Base | ~70-75% | 100 | ~24h |

### ImageNet-1K (1000 classes)

| Model | Top-1 Acc | Top-5 Acc |
|-------|-----------|-----------|
| Swin-Tiny | 81.3% | 95.5% |
| Swin-Small | 83.0% | 96.2% |
| Swin-Base | 83.5% | 96.5% |

## Troubleshooting

### Training Doesn't Converge

1. Check data loading (visualize samples)
2. Verify normalization statistics
3. Try lower learning rate
4. Ensure proper weight initialization

### Low Accuracy

1. Train longer
2. Add data augmentation
3. Use larger model
4. Tune hyperparameters

### Memory Issues

1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision
4. Use smaller window size
