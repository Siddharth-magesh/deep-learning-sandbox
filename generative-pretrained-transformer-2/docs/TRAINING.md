# Training Guide

## Training Pipeline

### Overview

The training pipeline includes:
- Data loading and preprocessing
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpoint management
- TensorBoard logging

## Data Preparation

### Dataset: WikiText-2

Default dataset for training:
- **Train**: 36,718 samples
- **Validation**: 3,760 samples  
- **Test**: 4,358 samples

### Tokenization

Uses GPT-2 tokenizer from HuggingFace:
- Vocabulary size: 50,257
- BPE (Byte-Pair Encoding)
- Max sequence length: 1024 tokens

### Data Processing

1. Load raw text from dataset
2. Tokenize with GPT-2 tokenizer
3. Pad/truncate to max_length
4. Create labels (shifted input_ids)
5. Apply attention masking

## Training Configuration

### Optimizer: AdamW

```
Learning Rate: 3e-4
Weight Decay: 0.01
Beta1: 0.9
Beta2: 0.95
Epsilon: 1e-8
```

AdamW decouples weight decay from gradient updates for better regularization.

### Learning Rate Schedule

**Warmup + Cosine Decay:**

1. **Warmup Phase** (0 to warmup_steps):
   - Linear increase from 0 to peak learning rate
   - Duration: 2,000 steps
   - Helps stabilize early training

2. **Cosine Decay Phase** (warmup_steps to end):
   - Smooth decrease following cosine curve
   - Minimum LR: 0
   - Better convergence than step decay

**Formula:**
```
if step < warmup_steps:
    lr = peak_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = peak_lr * 0.5 * (1 + cos(π * progress))
```

### Gradient Management

**Gradient Clipping:**
- Max norm: 1.0
- Prevents exploding gradients
- Stabilizes training

**Gradient Accumulation:**
- Accumulation steps: 1 (default)
- Effective batch size = batch_size × accumulation_steps
- Use for larger effective batches with limited memory

### Mixed Precision Training

**Automatic Mixed Precision (AMP):**
- FP16 for forward/backward pass
- FP32 for optimizer updates
- 2-3x faster training
- Reduces memory usage by ~40%

**Loss Scaling:**
- Dynamic loss scaling to prevent underflow
- Automatic adjustment during training

## Training Process

### Time-Based Training Limits

**Maximum Training Time:**
- Default: 12.0 hours
- Configurable via `max_training_hours` parameter
- Training stops automatically when time limit is reached
- Saves checkpoint before exiting

**Time Tracking:**
- Elapsed time displayed after each epoch
- Remaining time calculated dynamically
- Format: `Elapsed: X.XXh | Remaining: Y.YYh`

**Usage:**
```bash
# Train for maximum 8 hours
python -m generative-pretrained-transformer-2.src.main train --max_training_hours 8.0

# Train for maximum 2 hours (quick test)
python -m generative-pretrained-transformer-2.src.main train --max_training_hours 2.0
```

### Epoch Loop

```
For each epoch:
    1. Check elapsed time vs. max_training_hours
    2. Exit if time limit reached (save checkpoint)
    3. Set model to train mode
    4. Iterate over training batches
    5. Forward pass (compute loss)
    6. Backward pass (compute gradients)
    7. Gradient clipping
    8. Optimizer step
    9. Learning rate update
    10. Periodic validation
    11. Checkpoint saving
    12. Display elapsed and remaining time
```

### Batch Processing

```
For each batch:
    - Load input_ids and labels
    - Move to GPU/device
    - Compute logits and loss
    - Scale loss by accumulation_steps
    - Backpropagate
    - Accumulate gradients
    - Update weights (every accumulation_steps)
```

### Loss Function

**Cross-Entropy Loss:**
```
Loss = -∑ log(P(x_t | x_<t))
```

- Averaged over non-padded tokens
- Ignore index: -100 (padding tokens)
- Measures prediction quality

### Validation

**Validation Frequency:**
- Every 500 steps (configurable)
- After each epoch

**Validation Process:**
1. Set model to eval mode
2. Disable gradient computation
3. Compute loss on validation set
4. Calculate perplexity
5. Save best model
6. Resume training

## Checkpointing

### Checkpoint Contents

Each checkpoint includes:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Scaler state dict (if using AMP)
- Current epoch
- Global step
- Best validation loss
- Model configuration

### Checkpoint Strategy

**During Training:**
- Save every 1,000 steps
- Save after each epoch
- Save best model (lowest validation loss)

**Checkpoint Files:**
```
checkpoints/
├── best_model.pth              # Best validation loss
├── checkpoint_epoch_1.pth      # End of epoch 1
├── checkpoint_epoch_2.pth      # End of epoch 2
├── checkpoint_step_1000.pth    # Step 1000
├── checkpoint_step_2000.pth    # Step 2000
└── checkpoint_time_limit.pth   # Saved when max_training_hours reached
```

### Resume Training

Load checkpoint and continue:
```bash
python -m generative-pretrained-transformer-2.src.main train \
    --resume_from checkpoints/checkpoint_epoch_5.pth
```

## Monitoring with TensorBoard

### Logged Metrics

**Training Metrics:**
- Loss (every 100 steps)
- Learning rate (every 100 steps)
- Perplexity (every 100 steps)

**Validation Metrics:**
- Loss (every 500 steps)
- Perplexity (every 500 steps)

### Launch TensorBoard

```bash
cd d:\ai_research_learning
tensorboard --logdir=generative-pretrained-transformer-2/runs
```

Access at: http://localhost:6006

### Metrics to Monitor

**Loss:**
- Should decrease steadily
- Training loss should be lower than validation
- Large gap indicates overfitting

**Perplexity:**
- exp(loss)
- Lower is better
- Measures model uncertainty
- Target: <50 for WikiText-2

**Learning Rate:**
- Should follow warmup + cosine schedule
- Verify schedule is working correctly

## Training Commands

### Basic Training

```bash
python -m generative-pretrained-transformer-2.src.main train
```

### Time-Limited Training

```bash
# Train for maximum 8 hours
python -m generative-pretrained-transformer-2.src.main train --max_training_hours 8.0

# Train overnight (10 hours)
python -m generative-pretrained-transformer-2.src.main train --max_training_hours 10.0
```

### Custom Configuration

```bash
python -m generative-pretrained-transformer-2.src.main train \
    --max_epochs 20 \
    --max_training_hours 12.0 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --d_model 1024 \
    --num_layers 16 \
    --checkpoint_dir my_checkpoints
```

### Small Model (for testing)

```bash
python -m generative-pretrained-transformer-2.src.main train \
    --d_model 512 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 2048 \
    --batch_size 16
```

### Large Model (more capacity)

```bash
python -m generative-pretrained-transformer-2.src.main train \
    --d_model 1024 \
    --num_layers 24 \
    --num_heads 16 \
    --d_ff 4096 \
    --batch_size 4
```

## Training Tips

### Memory Optimization

**If OOM (Out of Memory):**
1. Reduce batch_size
2. Reduce max_length
3. Use gradient_accumulation
4. Reduce model size (d_model, num_layers)
5. Enable gradient checkpointing

### Speed Optimization

**Faster Training:**
1. Increase batch_size (if memory allows)
2. Use mixed precision (enabled by default)
3. Increase num_workers for data loading
4. Use pin_memory=True
5. Reduce eval_every frequency

### Stability

**If Training Unstable:**
1. Reduce learning rate
2. Increase warmup_steps
3. Reduce gradient_clip threshold
4. Check for NaN losses
5. Reduce dropout rates

## Expected Training Time

**Hardware:**
- NVIDIA GeForce RTX GPU
- 8GB VRAM

**Time Estimates:**
- Epoch duration: 15-20 minutes
- 10 epochs: 2.5-3.5 hours
- 50 epochs: 12-17 hours

**Faster with:**
- Higher-end GPU (RTX 4090, A100)
- Multiple GPUs (data parallel)
- Larger batch sizes

## Evaluation After Training

Check model quality:

```bash
python -m generative-pretrained-transformer-2.src.main evaluate \
    --model_path checkpoints/best_model.pth
```

Expected results after 10 epochs:
- Test Loss: 3.5-4.0
- Test Perplexity: 35-50

## Common Issues

### NaN Loss

**Causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability

**Solutions:**
- Reduce learning rate
- Increase warmup_steps
- Check gradient clipping
- Use mixed precision

### Slow Convergence

**Solutions:**
- Increase learning rate
- Reduce warmup_steps
- Increase batch size
- Check data quality

### Overfitting

**Signs:**
- Large train/val loss gap
- Val loss increases while train loss decreases

**Solutions:**
- Increase dropout
- Add weight decay
- Use more data
- Reduce model size
