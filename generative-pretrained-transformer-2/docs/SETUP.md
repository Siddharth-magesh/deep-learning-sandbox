# Quick Setup & Training Guide

## ✅ Optimized for Your System

**Your Hardware:**
- NVIDIA GeForce GPU (4GB dedicated VRAM)
- AMD Radeon Graphics
- 27.9 GB System RAM

**Optimized Settings:**
- `batch_size`: 4 (fits in 4GB VRAM)
- `accumulation_steps`: 2 (effective batch size = 8)
- `mixed_precision`: True (FP16 training)
- `max_epochs`: 50 (12-15 hours training)
- `num_workers`: 2 (optimized for your CPU)

## Installation

```powershell
cd d:\ai_research_learning
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install transformers datasets optuna tensorboard tqdm
```

## Training

### Start Training (50 Epochs - 12-15 Hours)

```powershell
cd d:\ai_research_learning
uv run python -m generative-pretrained-transformer-2.src.main train
```

**Or use the quick-start script:**
```powershell
cd d:\ai_research_learning
.\start_training.ps1
```

**What happens:**
- Trains for 50 epochs (~12-15 hours)
- Shows detailed model architecture summary
- Saves checkpoints every 500 steps
- Validates every 250 steps
- Logs to TensorBoard every 50 steps
- Auto-saves best model
- Uses your NVIDIA GPU automatically

### Quick Test (30 Minutes)
```powershell
uv run python -m generative-pretrained-transformer-2.src.main train --max_epochs 3
```

### Extended Training (24+ Hours, Maximum Accuracy)
```powershell
uv run python -m generative-pretrained-transformer-2.src.main train --max_epochs 100
```

### Resume Training (If Interrupted)
```powershell
uv run python -m generative-pretrained-transformer-2.src.main train --resume_from generative-pretrained-transformer-2/checkpoints/checkpoint_epoch_25.pth
```

### Custom Configuration
```powershell
uv run python -m generative-pretrained-transformer-2.src.main train --max_epochs 75 --batch_size 4 --learning_rate 3e-4
```

## Monitor Training

Open another PowerShell terminal:
```powershell
cd d:\ai_research_learning
uv run tensorboard --logdir=runs
```

Visit: http://localhost:6006

## Other Commands
Then visit: http://localhost:6006

## Other Commands

### Evaluate Model
```powershell
uv run python -m generative-pretrained-transformer-2.src.main evaluate --model_path generative-pretrained-transformer-2/checkpoints/best_model.pth
```

### Interactive Text Generation
```powershell
uv run python -m generative-pretrained-transformer-2.src.inference --model_path generative-pretrained-transformer-2/checkpoints/best_model.pth --interactive
```

### Single Prompt Generation
```powershell
uv run python -m generative-pretrained-transformer-2.src.inference --model_path generative-pretrained-transformer-2/checkpoints/best_model.pth --prompt "Once upon a time"
```

### Optimize Hyperparameters
```powershell
uv run ## Expected Performance

**Training Time:**
- 3 epochs: ~30 minutes
- 10 epochs: ~1.5-2 hours
- 50 epochs: ~12-15 hours ⭐ (Recommended for best accuracy)
- 100 epochs: ~24-30 hours (Maximum accuracy)

**Memory Usage:**
- Model: ~500MB
- Training: ~3.5GB VRAM (fits in 4GB)

**Expected Results:**

After 10 epochs:
- Validation Perplexity: ~40-50
- Loss: ~3.8-4.2

After 50 epochs (12-15 hours): ⭐
- Validation Perplexity: ~25-35
- Loss: ~3.2-3.6
- Much better text generation quality

After 100 epochs (24+ hours):
- Validation Perplexity: ~20-30
- Loss: ~3.0-3.4
- Best quality text generation

## Model Structure Summary

When you run training, you'll see:
- Total parameters breakdown
- Layer-by-layer architecture
- Memory estimates
- Parameter distribution (embeddings, attention, feed-forward)

## Pro Tips

1. **Monitor GPU usage**: Open Task Manager → Performance → GPU while training
2. **Check TensorBoard**: Watch loss decrease in real-time
3. **Save best model**: Auto-saved to `checkpoints/best_model.pth`
4. **Resume if needed**: Training will auto-resume from last checkpoint
5. **Test early**: Try inference after 10 epochs to see progress

All optimized for your 4GB NVIDIA GPU! 🚀
