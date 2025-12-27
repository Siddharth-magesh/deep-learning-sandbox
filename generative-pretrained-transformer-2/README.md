# GPT-2 Implementation from Scratch

A complete implementation of GPT-2 (Generative Pre-trained Transformer 2) with training, evaluation, hyperparameter optimization, and inference capabilities.

## Features

- **Full GPT-2 Architecture**: Multi-head attention, feed-forward networks, layer normalization
- **Training Pipeline**: Mixed precision training with gradient accumulation and checkpointing
- **TensorBoard Integration**: Real-time monitoring of training metrics
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Interactive Inference**: Streaming text generation with customizable parameters
- **Production Ready**: Proper configuration management and modular design

## Project Structure

```
generative-pretrained-transformer-2/
├── src/
│   ├── config/
│   │   ├── model_config.py        # GPT-2 model configuration
│   │   ├── train_config.py        # Training hyperparameters
│   │   ├── data_config.py         # Dataset configuration
│   │   ├── optuna_config.py       # Optimization settings
│   │   └── __init__.py
│   ├── data/
│   │   ├── dataset.py             # Dataset class
│   │   ├── datamodule.py          # Data loading utilities
│   │   └── __init__.py
│   ├── models/
│   │   ├── gpt2.py                # GPT-2 model implementation
│   │   └── __init__.py
│   ├── modules/
│   │   ├── attention.py           # Multi-head attention
│   │   ├── feedforward.py         # Feed-forward network
│   │   ├── decoder_block.py       # Transformer decoder block
│   │   └── __init__.py
│   ├── optim/
│   │   ├── optimizer.py           # Optuna hyperparameter optimization
│   │   ├── scheduler.py           # Learning rate scheduler
│   │   └── __init__.py
│   ├── utils/
│   │   ├── checkpoint.py          # Checkpoint utilities
│   │   ├── metrics.py             # Metric calculations
│   │   └── __init__.py
│   ├── main.py                    # CLI entry point
│   ├── train.py                   # Training logic
│   ├── evaluate.py                # Evaluation metrics
│   ├── inference.py               # Text generation
│   └── __init__.py
├── docs/
│   ├── ARCHITECTURE.md            # Model architecture details
│   ├── TRAINING.md                # Training guide
│   ├── INFERENCE.md               # Inference usage
│   └── API_REFERENCE.md           # API documentation
├── checkpoints/                   # Saved model checkpoints
├── optuna_results/                # Optimization results
├── runs/                          # TensorBoard logs
└── README.md
```

## Installation

Install dependencies using uv:

```bash
cd d:\ai_research_learning
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install transformers datasets optuna tensorboard tqdm
```

## Quick Start

### Training

Train the model on WikiText-2 dataset:

```bash
cd d:\ai_research_learning
python -m generative-pretrained-transformer-2.src.main train --max_epochs 10
```

### Evaluation

Evaluate a trained model:

```bash
python -m generative-pretrained-transformer-2.src.main evaluate --model_path generative-pretrained-transformer-2/checkpoints/best_model.pth
```

### Hyperparameter Optimization

Find optimal hyperparameters with Optuna:

```bash
python -m generative-pretrained-transformer-2.src.main optimize --n_trials 50
```

### Inference

#### Interactive Mode

```bash
python -m generative-pretrained-transformer-2.src.inference --model_path generative-pretrained-transformer-2/checkpoints/best_model.pth --interactive
```

#### Single Prompt

```bash
python -m generative-pretrained-transformer-2.src.inference --model_path generative-pretrained-transformer-2/checkpoints/best_model.pth --prompt "Once upon a time"
```

## Configuration

### Model Configuration (GPT2Config)

- `vocab_size`: 50257 (GPT-2 vocabulary)
- `context_length`: 1024 tokens
- `d_model`: 768 (embedding dimension)
- `num_heads`: 12 (attention heads)
- `num_layers`: 12 (transformer blocks)
- `d_ff`: 3072 (feed-forward dimension)
- `dropout`: 0.1

### Training Configuration (TrainingConfig)

- `batch_size`: 8
- `learning_rate`: 3e-4
- `weight_decay`: 0.01
- `max_epochs`: 10
- `max_training_hours`: 12.0
- `warmup_steps`: 2000
- `gradient_clip`: 1.0
- `mixed_precision`: True

### Inference Configuration (InferenceConfig)

- `max_new_tokens`: 100
- `temperature`: 0.8 (sampling randomness)
- `top_k`: 50 (top-k sampling)
- `top_p`: 0.95 (nucleus sampling)
- `repetition_penalty`: 1.2
- `stream`: True (streaming output)

## TensorBoard Monitoring

View training progress:

```bash
cd d:\ai_research_learning
tensorboard --logdir=generative-pretrained-transformer-2/runs
```

Access at: http://localhost:6006

## Model Details

### Architecture

- **Type**: Decoder-only transformer
- **Parameters**: ~124M (default configuration)
- **Context Window**: 1024 tokens
- **Positional Encoding**: Learned embeddings
- **Attention**: Causal multi-head self-attention
- **Activation**: GELU
- **Normalization**: Layer normalization (pre-norm)

### Training Features

- Mixed precision training (FP16)
- Gradient accumulation
- Learning rate warmup with cosine decay
- Gradient clipping
- Automatic checkpointing
- Real-time TensorBoard logging

## Advanced Usage

### Resume Training

```bash
python -m generative-pretrained-transformer-2.src.main train --resume_from checkpoints/checkpoint_epoch_5.pth
```

### Time-Limited Training

```bash
python -m generative-pretrained-transformer-2.src.main train --max_training_hours 8.0
```

### Custom Dataset

```bash
python -m generative-pretrained-transformer-2.src.main train --dataset_name your_dataset --dataset_config your_config
```

### CPU Training

```bash
python -m generative-pretrained-transformer-2.src.main train --device cpu
```

### Custom Generation Parameters

```bash
python -m generative-pretrained-transformer-2.src.inference --model_path checkpoints/best_model.pth --prompt "In the future" --temperature 0.9 --top_k 40 --max_new_tokens 150
```

## Documentation

- [Architecture Details](docs/ARCHITECTURE.md) - Deep dive into model architecture
- [Training Guide](docs/TRAINING.md) - Comprehensive training documentation
- [Inference Guide](docs/INFERENCE.md) - Text generation options and usage
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation

## License

MIT License

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
