# ResNet100 Architecture and Training Documentation

## Model Architecture

### Overview
ResNet100 is a deep convolutional neural network based on the Residual Network (ResNet) architecture with 100 layers. It uses bottleneck blocks with skip connections to enable training of very deep networks.

### Architecture Details

**Input Layer**
- Convolutional layer: 7x7 kernel, stride 2, 64 filters
- Batch Normalization
- ReLU activation
- Max pooling: 3x3 kernel, stride 2

**Residual Blocks (Bottleneck Structure)**

Each bottleneck block consists of:
1. 1x1 convolution (channel reduction)
2. 3x3 convolution (feature extraction)
3. 1x1 convolution (channel expansion by 4x)
4. Skip connection with identity mapping or 1x1 convolution for dimension matching

**Layer Configuration**
- Layer 1: 3 blocks, 64 base channels → 256 output channels
- Layer 2: 4 blocks, 128 base channels → 512 output channels
- Layer 3: 23 blocks, 256 base channels → 1024 output channels
- Layer 4: 3 blocks, 512 base channels → 2048 output channels

Total blocks: 3 + 4 + 23 + 3 = 33 bottleneck blocks

**Output Layer**
- Global average pooling
- Fully connected layer to num_classes

### Skip Connections
Skip connections are implemented using:
- Identity mapping when input/output dimensions match
- 1x1 convolution for dimension matching when stride ≠ 1 or channels don't match

### Weight Initialization
- Convolutional layers: Kaiming Normal initialization
- Batch Normalization: weights initialized to 1, biases to 0

## Dataset: EuroSAT

**Dataset Information**
- Source: Kaggle (apollo2506/eurosat-dataset)
- Classes: 10 land use and land cover categories
- Image format: RGB satellite imagery
- Original size: 64x64 pixels (can be resized)

**Classes**
1. Annual Crop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial
6. Pasture
7. Permanent Crop
8. Residential
9. River
10. Sea/Lake

**Data Splits**
- Training: 70%
- Validation: 15%
- Testing: 15%

## Training Pipeline

### Data Augmentation (Training)
- Resize to configured size (64, 128, or 224)
- Random horizontal flip
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation ±20%)
- Normalization with ImageNet statistics

### Validation/Test Transforms
- Resize to configured size
- Normalization with ImageNet statistics

### Optimization
- Optimizer: Adam
- Loss function: Cross-Entropy Loss
- Learning rate scheduling: StepLR, ReduceLROnPlateau, or CosineAnnealingLR

### Hyperparameter Optimization (Optuna)
Optimized parameters:
- Batch size: [8, 16, 32, 64]
- Learning rate: [1e-5, 1e-2] (log scale)
- Weight decay: [1e-6, 1e-3] (log scale)
- Image size: [64, 128, 224]
- Scheduler type: StepLR, ReduceLROnPlateau, CosineAnnealingLR

### Training Features
- **TensorBoard Logging**: Track loss, accuracy, and learning rate
- **Profiling**: PyTorch profiler for performance analysis
- **Mixed Precision Training**: Optional AMP for faster training on GPU
- **Checkpointing**: Save best model based on validation accuracy

## File Structure

```
residual-network/
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   ├── data_loader.py         # Dataset download and DataLoader creation
│   ├── resnet100.py           # ResNet100 model definition
│   ├── train.py               # Training loop with TensorBoard
│   ├── evaluate.py            # Model evaluation and metrics
│   ├── optimize.py            # Optuna hyperparameter optimization
│   ├── main.py                # Main training script
│   └── modules/
│       ├── bottle_neck.py     # Bottleneck block implementation
│       └── __init__.py
├── docs/
│   └── ARCHITECTURE.md        # This file
├── checkpoints/               # Model checkpoints (auto-created)
├── runs/resnet100/            # TensorBoard logs (auto-created)
├── profiler_logs/             # Profiler data (auto-created)
└── optuna_results/            # Optuna results (auto-created)
```

## Usage

### 1. Hyperparameter Optimization
```bash
cd residual-network/src
python optimize.py
```
Results saved to `residual-network/optuna_results/`

### 2. Training
```bash
cd residual-network/src
python main.py
```
- Model checkpoints: `residual-network/checkpoints/`
- TensorBoard logs: `residual-network/runs/resnet100/`
- Profiler data: `residual-network/profiler_logs/`

### 3. View TensorBoard
```bash
tensorboard --logdir=residual-network/runs
```

### 4. View Profiler Data
```bash
tensorboard --logdir=residual-network/profiler_logs
```

## Model Output

### Training Metrics
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Learning rate tracking

### Evaluation Metrics
- Test accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Per-class classification report
- Confusion matrix

### Saved Files
- `best_model.pth`: Best model checkpoint with state dicts
- `optuna_results.csv`: All Optuna trial results
- `best_hyperparameters.txt`: Best hyperparameters from Optuna

## Configuration Parameters

Key parameters in `config.py`:
- `num_classes`: Number of output classes (10 for EuroSAT)
- `img_size`: Input image size (64, 128, or 224)
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `weight_decay`: L2 regularization
- `use_amp`: Mixed precision training (GPU only)
- `use_scheduler`: Enable learning rate scheduling
- `save_best_only`: Save only best model or all checkpoints

## Performance Tips

**For Laptop/CPU Training**
- Use smaller batch size (8-16)
- Reduce image size to 64
- Disable AMP and profiler
- Fewer epochs for testing

**For GPU Training**
- Increase batch size (32-64)
- Use image size 128 or 224
- Enable AMP for faster training
- Use learning rate scheduling
- Enable profiler for bottleneck analysis
