# CLIP (Contrastive Language-Image Pretraining) - Implementation from Scratch

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Training](#training)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [References](#references)

## 🎯 Overview

This is a complete implementation of CLIP (Contrastive Language-Image Pretraining) from scratch using PyTorch. CLIP learns visual concepts from natural language supervision by training an image encoder and text encoder to predict which image-text pairs are correct.

**Key Features:**
- ✅ Vision Transformer (ViT) for image encoding
- ✅ Text Transformer for caption encoding
- ✅ Contrastive learning with temperature scaling
- ✅ Training on Flickr30k dataset
- ✅ Hyperparameter optimization with Optuna
- ✅ TensorBoard logging
- ✅ Checkpoint management

## 🏗️ Architecture

CLIP consists of two encoders that project images and text into a shared embedding space:

```
Input Image (224×224×3)           Input Text (77 tokens)
        ↓                                  ↓
  Vision Transformer              Text Transformer
   (Patch Embedding)              (Token Embedding)
        ↓                                  ↓
   12 Transformer                  8 Transformer
      Blocks                          Blocks
        ↓                                  ↓
 Image Features (512-D)          Text Features (512-D)
        ↓                                  ↓
        └──────────> Contrastive Loss <────┘
                   (Temperature Scaled)
```

### Vision Transformer
- **Input**: 224×224 RGB images
- **Patch Size**: 16×16 (196 patches)
- **Embedding Dim**: 768
- **Depth**: 12 transformer blocks
- **Output**: 512-D feature vector

### Text Transformer
- **Input**: 77 tokens (max length)
- **Vocab Size**: 49,408
- **Embedding Dim**: 512
- **Depth**: 8 transformer blocks
- **Output**: 512-D feature vector

### Contrastive Loss
- Computes similarity between all image-text pairs in a batch
- Uses temperature scaling (default: 0.07)
- Symmetric cross-entropy loss (image-to-text + text-to-image)

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+
pip install torch torchvision
pip install pandas numpy pillow tqdm
pip install kagglehub
pip install optuna tensorboard
```

### Dataset Setup
The Flickr30k dataset is automatically downloaded via KaggleHub:
```python
# Automatic download on first run
# Dataset cached at: ~/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/
```

**Dataset Statistics:**
- 31,785 images
- 158,925 image-caption pairs (5 captions per image)
- Downloaded size: ~5GB

## 📖 Usage

### 1. Training from Scratch

```bash
# Train with default configuration
python src/main.py
```

**Default Configuration:**
- Batch size: 64
- Learning rate: 1e-4
- Epochs: 20
- Weight decay: 1e-4
- Device: Auto-detect (CUDA/CPU)

### 2. Custom Training

Edit `src/config.py` to modify hyperparameters:

```python
config = Config()
config.batch_size = 128
config.learning_rate = 5e-4
config.num_epochs = 30
```

### 3. Training with Limited Data (Debugging)

```python
# In main.py
trainer.load_data(max_samples=1000)  # Use only 1000 samples
```

### 4. Resume from Checkpoint

```python
trainer.build_model()
trainer.load_checkpoint('checkpoints/checkpoint_epoch_10.pth')
trainer.train()
```

### 5. Hyperparameter Optimization

```bash
# Run Optuna optimization (10 trials)
python src/optimize.py
```

Results saved to:
- `optuna-results/best_hyperparameters.txt`
- `optuna-results/best_results.json`
- `optuna-results/optimization_history.png`
- `optuna-results/param_importances.png`

## 📁 Project Structure

```
contrastive-language-image-pretraining/
├── src/
│   ├── clip.py                 # CLIP model + loss
│   ├── config.py               # Hyperparameters
│   ├── data_loader.py          # Flickr30k dataset
│   ├── train.py                # Trainer class
│   ├── main.py                 # Training script
│   ├── optimize.py             # Optuna optimization
│   ├── vision_transformer.py   # Vision encoder
│   ├── text_transformer.py     # Text encoder
│   ├── evaluate.py             # Evaluation metrics
│   └── modules/
│       ├── transformer.py       # Transformer block
│       ├── multi_head_attention.py
│       ├── multi_layer_perceptron.py
│       └── patch_embedding.py
├── docs/
│   ├── README.md               # This file
│   ├── ARCHITECTURE.md         # Detailed architecture
│   ├── TRAINING_GUIDE.md       # Training guide
│   └── API_REFERENCE.md        # Code reference
├── checkpoints/                # Saved models
├── optuna-results/             # Optimization results
└── runs/                       # TensorBoard logs
```

## 🔑 Key Components

### 1. **CLIP Model** (`src/clip.py`)
Main model combining vision and text encoders.

```python
model = CLIP(
    vision_embed_dim=768,
    text_embed_dim=512,
    output_dim=512,
    temperature=0.07
)
```

### 2. **Vision Transformer** (`src/vision_transformer.py`)
Processes images into embeddings.

- Patch embedding: Divides image into 16×16 patches
- Positional encoding: Learnable position embeddings
- CLS token: Used for final representation

### 3. **Text Transformer** (`src/text_transformer.py`)
Processes text into embeddings.

- Token embedding: Maps tokens to vectors
- Positional encoding: Learnable position embeddings
- EOS token pooling: Uses end-of-sequence token

### 4. **Data Loader** (`src/data_loader.py`)
Handles Flickr30k dataset loading.

- Automatic dataset download
- Simple hash-based tokenizer
- Augmentation and normalization

### 5. **Trainer** (`src/train.py`)
Manages training loop.

- Progress tracking with tqdm
- Automatic checkpointing
- TensorBoard logging
- Learning rate scheduling

## 🎓 Training

### Training Process

1. **Data Loading**: Flickr30k images + captions
2. **Forward Pass**: Encode images and text
3. **Similarity Matrix**: Compute all pairwise similarities
4. **Contrastive Loss**: Maximize diagonal (correct pairs)
5. **Backpropagation**: Update both encoders
6. **Scheduler Step**: Adjust learning rate

### Monitoring Training

```bash
# View TensorBoard logs
tensorboard --logdir=runs/clip_training
```

### Checkpoints

Checkpoints are saved:
- Every 5 epochs
- When achieving best loss
- Contains: model, optimizer, scheduler states

**Loading checkpoint:**
```python
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint)
```

## ⚡ Hyperparameter Optimization

### Priority Levels

**HIGH PRIORITY** (biggest impact):
1. `learning_rate` [1e-5, 1e-3]
2. `temperature` [0.01, 0.1]
3. `batch_size` [32, 64, 128]
4. `weight_decay` [1e-6, 1e-3]

**MEDIUM PRIORITY**:
5. `text_depth` [6, 8, 12]
6. `depth` [10, 12, 14]
7. `embed_dim` [512, 768, 1024]
8. `dropout` [0.05, 0.1, 0.2]

**LOW PRIORITY**:
9. `num_heads` [8, 12, 16]
10. `mlp_ratio` [2.0, 4.0, 6.0]

### Running Optimization

```bash
python src/optimize.py
```

This will:
1. Train 10 different configurations
2. Track best performing setup
3. Generate visualization plots
4. Save results to `optuna-results/`

## 📚 References

- **Original Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **Dataset**: [Flickr30k Entities](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/)

## 🎯 Quick Start Commands

```bash
# 1. Train model
python src/main.py

# 2. Optimize hyperparameters
python src/optimize.py

# 3. View training progress
tensorboard --logdir=runs

# 4. Evaluate model (coming soon)
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

## 💡 Tips

1. **GPU Memory**: Reduce batch size if OOM errors occur
2. **Fast Debugging**: Use `max_samples=1000` during development
3. **Best Performance**: Train for 30-50 epochs on full dataset
4. **Learning Rate**: Start with 1e-4, adjust based on convergence
5. **Temperature**: Lower values (0.01-0.05) often work better

## 📞 Support

For issues, questions, or contributions, please refer to the detailed documentation in the `docs/` folder.
