# ResNet - Residual Network

Implementation of ResNet50 and ResNet100 for image classification using PyTorch.

## Project Structure

```
residual-network/
├── src/
│   ├── config.py           # Configuration settings
│   ├── data_loader.py      # Data loading utilities
│   ├── resnet.py          # ResNet50 and ResNet100 models
│   ├── main.py            # Main training script
│   ├── train.py           # Training logic
│   ├── evaluate.py        # Evaluation logic
│   ├── optimize.py        # Hyperparameter optimization with Optuna
│   └── modules/
│       └── bottle_neck.py # Bottleneck block implementation
├── docs/
│   ├── ARCHITECTURE.md    # Architecture documentation
│   └── DATA_LOADING.md    # Data loading documentation
├── checkpoints/           # Saved model checkpoints
├── runs/                  # TensorBoard logs
└── profiler_logs/        # PyTorch profiler outputs
```

## Installation

Install dependencies using uv:

```bash
uv pip install torch torchvision torchinfo datasets
```

## Usage

### 1. Train ResNet50 (Default)

```bash
cd D:\deep-learning-sandbox\residual-network\src
uv run main.py
```

### 2. Train ResNet100

Edit `config.py` and change:
```python
model_name: str = "resnet50"  # Change to "resnet100"
```

Or modify the config in your code:
```python
from config import Config

config = Config()
config.model_name = "resnet100"
```

Then run:
```bash
cd D:\deep-learning-sandbox\residual-network\src
uv run main.py
```

### 3. Hyperparameter Optimization with Optuna

```bash
cd D:\deep-learning-sandbox\residual-network\src
uv run optimize.py
```

### 4. View Training Progress with TensorBoard

```bash
tensorboard --logdir=D:\deep-learning-sandbox\residual-network\runs
```

Then open your browser to `http://localhost:6006`

### 5. View Profiler Results

```bash
tensorboard --logdir=D:\deep-learning-sandbox\residual-network\profiler_logs
```

## Models

### ResNet50
- Layers: [3, 4, 6, 3] bottleneck blocks
- Parameters: ~25.5M
- Suitable for: Medium-sized datasets

### ResNet100
- Layers: [3, 4, 23, 3] bottleneck blocks
- Parameters: ~44.5M
- Suitable for: Larger datasets, higher accuracy requirements

## Configuration

Key configuration options in `config.py`:

```python
model_name: str = "resnet50"        # Model architecture: "resnet50" or "resnet100"
batch_size: int = 8                 # Batch size for training
num_epochs: int = 5                 # Number of training epochs
learning_rate: float = 0.001        # Learning rate
img_size: int = 64                  # Input image size
device: str = "cuda"                # Device: "cuda" or "cpu"
use_amp: bool = False               # Use automatic mixed precision
use_scheduler: bool = False         # Use learning rate scheduler
```

## Quick Commands

All commands should be run from the project root or src directory:

```bash
# Train with default settings (ResNet50)
cd D:\deep-learning-sandbox\residual-network\src
uv run main.py

# Run hyperparameter optimization
uv run optimize.py

# View TensorBoard
tensorboard --logdir=../runs

# View profiler
tensorboard --logdir=../profiler_logs
```

## Output Structure

After training, you'll find:

```
residual-network/
├── checkpoints/
│   └── best_model.pth              # Best model checkpoint
├── runs/
│   ├── resnet50/                   # TensorBoard logs for ResNet50
│   └── resnet100/                  # TensorBoard logs for ResNet100
└── profiler_logs/                  # PyTorch profiler data
```

## Features

- ✓ ResNet50 and ResNet100 architectures
- ✓ Automatic mixed precision training
- ✓ Learning rate scheduling
- ✓ TensorBoard integration
- ✓ PyTorch profiler support
- ✓ Checkpoint saving/loading
- ✓ Hyperparameter optimization with Optuna
- ✓ Comprehensive evaluation metrics

## Dataset

Default dataset: EuroSAT (satellite imagery classification)
- 10 classes
- RGB images
- Downloaded automatically to `~/.cache/eurosat-dataset`

## Performance Monitoring

The training pipeline includes:
- Real-time metrics logging
- TensorBoard visualization
- Profiling for performance optimization
- Confusion matrix generation
- Per-class accuracy metrics
