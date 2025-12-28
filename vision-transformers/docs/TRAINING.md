# Training Guide

## Training Pipeline

### Data Preparation

CIFAR-10 dataset automatically downloaded and cached.

Training set: 50,000 images
Validation set: 10,000 images (using test set for validation)
Test set: 10,000 images

### Data Augmentation

Training transforms:
- RandomCrop(32, padding=4)
- RandomHorizontalFlip(p=0.5)
- ToTensor()
- Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

Validation transforms:
- ToTensor()
- Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

## Training Configuration

### Optimizer

AdamW optimizer with:
```
learning_rate: 3e-4
weight_decay: 0.01
beta1: 0.9
beta2: 0.999
epsilon: 1e-8
```

### Learning Rate Schedule

Warmup + Cosine Decay:

1. Warmup Phase (0 to 5 epochs):
   - Linear increase from 0 to peak LR
   - Stabilizes early training

2. Cosine Decay Phase (5 epochs to end):
   - Smooth decrease following cosine curve
   - Minimum LR: 0
   - Better final convergence

Formula:
```
if epoch < warmup_epochs:
    lr = peak_lr * (epoch / warmup_epochs)
else:
    progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
    lr = peak_lr * 0.5 * (1 + cos(π * progress))
```

### Regularization

Dropout:
- Attention dropout: 0.1
- MLP dropout: 0.1

Label Smoothing: 0.1

Gradient Clipping: 1.0

### Mixed Precision Training

FP16 automatic mixed precision:
- Forward/backward in FP16
- Optimizer updates in FP32
- 2x faster training
- 50% less memory

## Training Process

### Epoch Loop

```
For each epoch:
    1. Check time limit
    2. Train on all batches
    3. Update learning rate
    4. Validate (every eval_every epochs)
    5. Save checkpoint if best
    6. Save periodic checkpoint
```

### Batch Processing

```
For each batch:
    1. Load images and labels
    2. Forward pass (compute logits and loss)
    3. Backward pass (compute gradients)
    4. Clip gradients
    5. Optimizer step
    6. Scheduler step
    7. Log metrics
```

### Loss Function

Cross-entropy with label smoothing:
```
Loss = -sum(y_smooth * log(p))
```

Label smoothing smooths the target distribution:
```
y_smooth = (1 - α) * y_true + α / K
```
where α = 0.1, K = 10

## Checkpointing

### Checkpoint Contents

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Scaler state dict (if mixed precision)
- Current epoch
- Global step
- Best validation accuracy
- Model configuration

### Checkpoint Strategy

During training:
- Save every 5 epochs
- Save best model (highest validation accuracy)
- Save on time limit reached

Checkpoint files:
```
checkpoints/
├── best_model.pth
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
└── checkpoint_time_limit.pth
```

### Resume Training

```bash
uv run python -m vision-transformers.src.main train \\
    --resume_from checkpoints/checkpoint_epoch_50.pth \\
    --max_epochs 200
```

## Monitoring with TensorBoard

### Logged Metrics

Training (every 50 steps):
- Loss
- Accuracy
- Learning rate

Validation (every epoch):
- Loss
- Accuracy

### Launch TensorBoard

```bash
tensorboard --logdir=runs/vision_transformer
```

Access at: http://localhost:6006

### Metrics to Monitor

Loss:
- Should decrease steadily
- Training loss lower than validation

Accuracy:
- Should increase steadily
- Monitor for overfitting (large train/val gap)

Learning Rate:
- Verify warmup + cosine schedule
- Should start low, peak, then decay

## Training Commands

### Basic Training

```bash
uv run python -m vision-transformers.src.main train
```

### Custom Configuration

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

### Small Model (Faster Training)

```bash
uv run python -m vision-transformers.src.main train \\
    --embedding_dim 128 \\
    --num_layers 6 \\
    --num_heads 4 \\
    --mlp_size 512 \\
    --batch_size 256
```

### Large Model (Better Accuracy)

```bash
uv run python -m vision-transformers.src.main train \\
    --embedding_dim 384 \\
    --num_layers 24 \\
    --num_heads 6 \\
    --mlp_size 1536 \\
    --batch_size 64
```

## Training Tips

### Memory Optimization

If OOM (Out of Memory):
1. Reduce batch_size
2. Reduce embedding_dim
3. Reduce num_layers
4. Use gradient accumulation (manual implementation needed)

### Speed Optimization

Faster training:
1. Increase batch_size (if memory allows)
2. Use mixed precision (enabled by default)
3. Increase num_workers
4. Use pin_memory (enabled by default)

### Stability

If training unstable:
1. Reduce learning rate
2. Increase warmup_epochs
3. Reduce gradient_clip threshold
4. Check for NaN losses

## Expected Training Time

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU

Time estimates:
- 10 epochs: 30-40 minutes
- 50 epochs: 3-4 hours
- 100 epochs: 6-8 hours

Faster with:
- Higher-end GPU (RTX 4090, A100)
- Larger batch sizes
- Multiple GPUs

## Expected Results

After 100 epochs:
- Training Loss: 0.2-0.3
- Training Accuracy: 95-98%
- Validation Loss: 0.4-0.5
- Validation Accuracy: 85-90%

Per-class accuracy should be roughly balanced across all 10 classes.

## Common Issues

### Low Accuracy

Solutions:
- Increase num_layers or embedding_dim
- Train for more epochs
- Adjust learning rate
- Check data augmentation

### Overfitting

Signs:
- Large train/val accuracy gap
- Validation accuracy plateaus or decreases

Solutions:
- Increase dropout
- Stronger data augmentation
- Reduce model size
- Add weight decay

### Slow Convergence

Solutions:
- Increase learning rate
- Reduce warmup_epochs
- Adjust batch_size
- Check optimizer settings

### NaN Loss

Causes:
- Learning rate too high
- Gradient explosion
- Numerical instability

Solutions:
- Reduce learning rate
- Increase gradient clipping
- Check data normalization
- Use mixed precision
