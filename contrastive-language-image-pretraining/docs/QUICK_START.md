# 🚀 CLIP From Scratch - Quick Start Guide

## ✅ Fixed Issues

All code has been reviewed and the following issues were fixed:

1. ✅ **data_loader.py**: Fixed column name (' comment' with leading space)
2. ✅ **config.py**: Fixed typo ('pretraining' not 'prtraining')
3. ✅ **train.py**: Fixed enumerate bug in training loop
4. ✅ **main.py**: Removed premature model summary call
5. ✅ **train.py**: Removed duplicate setup_paths (handled by data_loader)

## 🎯 How to Start Training

### Option 1: Basic Training (Recommended for First Run)

```bash
cd d:\ai_research_learning\contrastive-language-image-pretraining

# Start training with default settings
python src\main.py
```

This will:
- ✅ Auto-download Flickr30k dataset (~5GB, first run only)
- ✅ Train for 20 epochs
- ✅ Save checkpoints every 5 epochs
- ✅ Log to TensorBoard
- ✅ Save best model automatically

**Expected time:** ~5-6 hours on GPU

### Option 2: Quick Test (Debugging)

```python
# Edit src/main.py, change this line:
trainer.load_data(max_samples=1000)  # Use only 1000 samples

# Then run:
python src\main.py
```

**Expected time:** ~30 minutes

### Option 3: Hyperparameter Optimization

```bash
# Find best hyperparameters automatically
python src\optimize.py
```

This will:
- Run 10 different configurations
- Save best hyperparameters
- Generate visualization plots
- Results in `optuna-results/`

**Expected time:** ~10-15 hours

## 📊 Monitor Training

### TensorBoard (Real-time)

```bash
# In a separate terminal
tensorboard --logdir=runs\clip_training

# Open browser: http://localhost:6006
```

### Console Output

```
Training: 100%|████████| 2481/2481 [15:32<00:00, loss: 1.8234, lr: 0.000098]
Epoch 1/20 | Loss: 2.4123 | Time: 932.45s
Checkpoint saved to checkpoints/checkpoint_epoch_1.pth
```

## 📁 Project Structure

```
contrastive-language-image-pretraining/
├── src/
│   ├── main.py          # ← START HERE (training entry point)
│   ├── optimize.py      # ← Hyperparameter optimization
│   ├── clip.py          # CLIP model
│   ├── train.py         # Trainer class
│   ├── config.py        # Configuration
│   ├── data_loader.py   # Dataset
│   ├── vision_transformer.py
│   ├── text_transformer.py
│   └── modules/
├── docs/
│   ├── README.md           # Full documentation
│   ├── ARCHITECTURE.md     # Model architecture
│   ├── TRAINING_GUIDE.md   # Detailed training guide
│   └── API_REFERENCE.md    # Code reference
├── checkpoints/         # Saved models (auto-created)
├── optuna-results/      # Optimization results (auto-created)
└── runs/               # TensorBoard logs (auto-created)
```

## 🔧 Configuration

### Default Settings (in `src/config.py`)

```python
# Model
embed_dim = 768          # Vision embedding dimension
depth = 12               # Vision transformer layers
text_embed_dim = 512     # Text embedding dimension
text_depth = 8           # Text transformer layers

# Training
batch_size = 64          # Batch size
num_epochs = 20          # Training epochs
learning_rate = 1e-4     # Learning rate
temperature = 0.07       # Contrastive loss temperature

# System
device = "cuda"          # Auto-detect GPU/CPU
num_workers = 4          # DataLoader workers
```

### Modify Configuration

**Option A: Edit `src/config.py` directly**

```python
@dataclass
class Config:
    batch_size: int = 128      # Change this
    learning_rate: float = 5e-4  # And this
```

**Option B: Override in code**

```python
# In main.py
config = Config()
config.batch_size = 128
config.num_epochs = 30
```

## 🎯 Common Use Cases

### 1. First-Time Training

```bash
# Just run it!
python src\main.py
```

Dataset downloads automatically, training starts.

### 2. Resume from Checkpoint

```python
# In main.py, add after trainer.build_model():
trainer.load_checkpoint('checkpoints/checkpoint_epoch_10.pth')
trainer.train()
```

### 3. Train on Limited Data (Testing)

```python
# In main.py:
trainer.load_data(max_samples=5000)
```

### 4. Find Best Hyperparameters

```bash
python src\optimize.py
```

Check results in `optuna-results/best_hyperparameters.txt`

### 5. Custom Model Size

```python
# In config.py:
embed_dim = 1024        # Larger model
depth = 14
batch_size = 32         # Reduce if OOM
```

## 📊 Expected Results

### Training Loss

```
Epoch 1:  Loss ~3.0-3.5   (high initial loss)
Epoch 5:  Loss ~2.0-2.5   (rapid improvement)
Epoch 10: Loss ~1.6-2.0   (steady progress)
Epoch 20: Loss ~1.4-1.8   (convergence)
```

### Checkpoints Saved

```
checkpoints/
├── checkpoint_epoch_1.pth
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_15.pth
├── checkpoint_epoch_20.pth
└── best_model.pth        # Best performing model
```

## 🚨 Troubleshooting

### GPU Out of Memory

```python
# Reduce batch size in config.py
batch_size = 32  # or 16
```

### Slow Training

```bash
# Check GPU usage
nvidia-smi

# Verify using GPU in output:
# "Using device: cuda"
```

### Dataset Not Found

The dataset downloads automatically on first run. If it fails:

```python
# Check internet connection
# Or manually download from Kaggle:
# https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
```

### Import Errors

```bash
# Install all dependencies
pip install torch torchvision pandas numpy pillow tqdm kagglehub optuna tensorboard
```

## 📚 Documentation

All documentation is in the `docs/` folder:

1. **[README.md](docs/README.md)** - Complete overview and usage
2. **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed model architecture
3. **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - In-depth training guide
4. **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Code reference

## 🎓 Next Steps

1. **Start Training**: `python src\main.py`
2. **Monitor Progress**: Open TensorBoard
3. **Wait for Completion**: ~5-6 hours
4. **Check Best Model**: `checkpoints/best_model.pth`
5. **Optimize (Optional)**: `python src\optimize.py`

## 💡 Tips

- **First run**: Test with `max_samples=1000` to verify everything works
- **GPU Memory**: Reduce batch size if you get OOM errors
- **Best Performance**: Use largest batch size that fits in memory
- **Learning Rate**: 1e-4 is a good starting point
- **Checkpoints**: Don't delete - you can resume from any epoch

## ✨ Key Features

✅ **Complete implementation** - All code from scratch  
✅ **Auto dataset download** - No manual setup  
✅ **Checkpoint management** - Resume training anytime  
✅ **TensorBoard logging** - Visual monitoring  
✅ **Hyperparameter optimization** - Automated tuning  
✅ **Well documented** - Comprehensive docs  
✅ **Production ready** - Clean, modular code  

## 🎯 Quick Commands Reference

```bash
# Basic training
python src\main.py

# Hyperparameter optimization  
python src\optimize.py

# TensorBoard
tensorboard --logdir=runs

# Check GPU
nvidia-smi
```

---

**That's it! You're ready to train CLIP from scratch. Good luck! 🚀**
