# Training Guide - CLIP Implementation

## 🎯 Quick Start

### Basic Training

```bash
# Navigate to project directory
cd ai_research_learning/contrastive-language-image-pretraining

# Start training with default settings
python src/main.py
```

### Expected Output

```
Configuration:
  img_size: 224
  patch_size: 16
  embed_dim: 768
  depth: 12
  batch_size: 64
  learning_rate: 0.0001
  num_epochs: 20
  ...

Using device: cuda

Loading dataset...
Loaded 31785 image-caption pairs from 31785 unique images
Loaded 2481 batches

Initializing CLIP model...
Model moved to cuda

Starting training for 20 epochs...

Training: 100%|███████████| 2481/2481 [15:32<00:00, loss: 2.3456, lr: 0.000098]
Epoch 1/20 | Loss: 2.4123 | Time: 932.45s
Checkpoint saved to checkpoints/checkpoint_epoch_1.pth

...
```

## 📝 Step-by-Step Training Process

### 1. Environment Setup

```bash
# Install dependencies
pip install torch torchvision pandas numpy pillow tqdm kagglehub optuna tensorboard

# Verify CUDA availability (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Configuration

Edit `src/config.py` to customize training:

```python
@dataclass
class Config:
    # Model architecture
    embed_dim: int = 768           # Vision transformer embedding
    depth: int = 12                # Number of ViT layers
    num_heads: int = 12            # Attention heads
    text_embed_dim: int = 512      # Text transformer embedding
    text_depth: int = 8            # Number of text layers
    
    # Training settings
    batch_size: int = 64           # Batch size (adjust for GPU memory)
    num_epochs: int = 20           # Training epochs
    learning_rate: float = 1e-4    # Initial learning rate
    weight_decay: float = 1e-4     # L2 regularization
    
    # Contrastive learning
    temperature: float = 0.07      # Temperature scaling
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4           # DataLoader workers
    save_dir: str = "./checkpoints"
```

### 3. Dataset Preparation

**Option A: Automatic (Recommended)**
```python
# Dataset auto-downloads on first run via KaggleHub
# Location: ~/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/
```

**Option B: Manual**
```python
# If you have dataset locally, modify data_loader.py:
def load_kaggle_flickr30k():
    image_dir = "/path/to/flickr30k_images"
    caption_file = "/path/to/results.csv"
    return image_dir, caption_file
```

### 4. Start Training

```bash
# Full training
python src/main.py

# Debug mode (limited samples)
# Edit main.py: trainer.load_data(max_samples=1000)
python src/main.py
```

### 5. Monitor Training

**TensorBoard (Real-time)**
```bash
# In a separate terminal
tensorboard --logdir=runs/clip_training

# Open browser to: http://localhost:6006
```

**Console Output**
```
Epoch 5/20 | Loss: 1.8234 | Time: 928.12s
Checkpoint saved to checkpoints/checkpoint_epoch_5.pth
Best model updated with loss: 1.8234
```

## ⚙️ Training Configurations

### Configuration 1: Fast Training (Debugging)

```python
# config.py
batch_size: int = 32
num_epochs: int = 5
depth: int = 6          # Reduced layers
text_depth: int = 4
```

```python
# main.py
trainer.load_data(max_samples=5000)  # Limited data
```

**Use when:**
- Testing code changes
- Quick experiments
- Limited GPU memory

**Expected time:** ~30 minutes

### Configuration 2: Standard Training

```python
# config.py (default values)
batch_size: int = 64
num_epochs: int = 20
learning_rate: float = 1e-4
```

**Use when:**
- Normal training runs
- Good GPU (8GB+ VRAM)

**Expected time:** ~5-6 hours

### Configuration 3: High-Performance Training

```python
# config.py
batch_size: int = 128         # Larger batches
num_epochs: int = 50          # More epochs
learning_rate: float = 5e-4   # Higher LR
embed_dim: int = 1024         # Larger model
depth: int = 14
```

**Use when:**
- High-end GPU (16GB+ VRAM)
- Production model
- Maximum performance

**Expected time:** ~20-24 hours

### Configuration 4: CPU Training

```python
# config.py
batch_size: int = 8           # Small batches
num_workers: int = 0          # No parallel loading
device: str = "cpu"
```

```python
# main.py
trainer.load_data(max_samples=1000)
```

**Use when:**
- No GPU available
- Testing only

**Expected time:** Very slow (~10x slower)

## 🎛️ Hyperparameter Tuning

### Learning Rate

```python
# Lower LR: More stable, slower convergence
learning_rate: float = 1e-5

# Default LR: Good balance
learning_rate: float = 1e-4

# Higher LR: Faster, may diverge
learning_rate: float = 5e-4
```

**Signs of wrong LR:**
- Too high: Loss diverges or NaN
- Too low: Very slow improvement

### Batch Size

```python
# Smaller batches
batch_size: int = 32
# Pros: Less memory, faster iterations
# Cons: Fewer negative samples, noisy gradients

# Larger batches
batch_size: int = 128
# Pros: More negative samples, stable gradients
# Cons: More memory, slower iterations
```

**Rule of thumb:** Largest batch that fits in memory

### Temperature

```python
# Lower temperature: Sharper distributions
temperature: float = 0.01

# Default
temperature: float = 0.07

# Higher temperature: Softer distributions
temperature: float = 0.10
```

**Effect:** Controls how "confident" the model should be

### Model Size

```python
# Smaller model (faster, less accurate)
embed_dim: int = 512
depth: int = 6
text_depth: int = 4

# Default
embed_dim: int = 768
depth: int = 12
text_depth: int = 8

# Larger model (slower, more accurate)
embed_dim: int = 1024
depth: int = 14
text_depth: int = 12
```

## 🔄 Resume Training

### From Checkpoint

```python
# In main.py
trainer = Trainer(config)
trainer.load_data()
trainer.build_model()

# Load checkpoint
checkpoint_path = "checkpoints/checkpoint_epoch_10.pth"
start_epoch = trainer.load_checkpoint(checkpoint_path)

# Continue training
trainer.train()
```

### Modify Configuration

```python
# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint_epoch_10.pth")

# Change learning rate
config.learning_rate = 5e-5

# Build new optimizer
trainer.build_model()
trainer.model.load_state_dict(checkpoint['model_state_dict'])

# Continue with new LR
trainer.train()
```

## 📊 Tracking Progress

### What to Monitor

1. **Training Loss**
   - Should decrease steadily
   - Typical range: 3.0 → 1.5

2. **Learning Rate**
   - Decreases with cosine schedule
   - Check if too high/low

3. **Epoch Time**
   - Should be consistent
   - Sudden slowdown = issue

4. **GPU Memory**
   - Monitor with `nvidia-smi`
   - Adjust batch size if OOM

### TensorBoard Metrics

```python
# Logged automatically
- Loss/train: Training loss per epoch
- Learning rate: Current LR
```

### Expected Loss Curve

```
Epoch  | Loss   | Expected Behavior
-------|--------|------------------
1      | 3.2    | High initial loss
5      | 2.4    | Rapid decrease
10     | 1.9    | Slowing down
20     | 1.6    | Near convergence
50     | 1.4    | Minimal improvement
```

## 🚨 Troubleshooting

### Issue: CUDA Out of Memory

```python
# Solution 1: Reduce batch size
batch_size: int = 32  # or 16

# Solution 2: Reduce model size
embed_dim: int = 512
depth: int = 6

# Solution 3: Use gradient accumulation
# (Not implemented, but can be added)
```

### Issue: Loss Not Decreasing

```python
# Check 1: Learning rate too low
learning_rate: float = 5e-4  # Increase

# Check 2: Data loading issue
# Verify: print(images.shape, captions.shape)

# Check 3: Model initialization
# Try: Different random seed
```

### Issue: Loss Exploding (NaN)

```python
# Solution 1: Lower learning rate
learning_rate: float = 1e-5

# Solution 2: Add gradient clipping
# In train.py, add:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Solution 3: Check temperature
temperature: float = 0.07  # Not too small
```

### Issue: Very Slow Training

```python
# Check 1: CPU/GPU usage
device: str = "cuda"  # Make sure using GPU

# Check 2: DataLoader workers
num_workers: int = 4  # Parallel data loading

# Check 3: Dataset size
# Use max_samples for debugging
trainer.load_data(max_samples=5000)
```

### Issue: Checkpoints Not Saving

```python
# Check save directory
save_dir: str = "./checkpoints"

# Ensure directory exists
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Check disk space
# Checkpoint size: ~600MB each
```

## 📈 Optimization with Optuna

### Run Hyperparameter Search

```bash
python src/optimize.py
```

### Customize Search Space

```python
# In optimize.py
def objective(trial):
    # Modify ranges
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    temperature = trial.suggest_float("temperature", 0.01, 0.15)
    
    # Add new parameters
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
```

### View Optimization Results

```bash
# Results location
optuna-results/
├── best_hyperparameters.txt
├── best_results.json
├── optimization_history.png
├── param_importances.png
└── parallel_coordinate.png
```

## 💾 Checkpoint Management

### Checkpoint Structure

```python
checkpoint = {
    'epoch': int,                    # Epoch number
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # LR scheduler state
    'loss': float,                   # Best loss
    'config': Config                 # Training config
}
```

### Load Specific Components

```python
# Load only model weights
model.load_state_dict(checkpoint['model_state_dict'])

# Load optimizer state
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Load scheduler
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

## 🎯 Best Practices

1. **Start Small**: Test with small dataset first
2. **Monitor Actively**: Watch loss and GPU memory
3. **Save Often**: Checkpoints every 5 epochs
4. **Use TensorBoard**: Visual monitoring is crucial
5. **Experiment**: Try different hyperparameters
6. **Document**: Keep notes on what works
7. **GPU Warmup**: First epoch may be slower
8. **Reproducibility**: Set random seeds

## 📚 Advanced Topics

### Custom Learning Rate Schedule

```python
# In train.py, modify build_model()
self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
    self.optimizer,
    max_lr=config.learning_rate * 10,
    epochs=config.num_epochs,
    steps_per_epoch=len(self.train_loader)
)
```

### Mixed Precision Training

```python
# Use AMP for faster training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    logits, _, _ = model(images, captions)
    loss = loss_fn(logits)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Training

```python
# For multiple GPUs (not implemented)
# Use torch.nn.DataParallel or DistributedDataParallel
model = nn.DataParallel(model)
```

## 🎓 Training Tips

1. **Warm Start**: First few epochs may have high loss
2. **Patience**: Good results take 20+ epochs
3. **Batch Size**: Larger = better (if memory allows)
4. **Learning Rate**: Most important hyperparameter
5. **Temperature**: Usually 0.07 works well
6. **Checkpoints**: Don't delete - disk is cheap
7. **Validation**: Add validation set for better tuning
8. **Data**: More data = better model

## 📞 Getting Help

If training issues persist:
1. Check error messages carefully
2. Verify dataset is loaded correctly
3. Test with smaller configuration
4. Review logs in `runs/clip_training`
5. Check GPU with `nvidia-smi`
