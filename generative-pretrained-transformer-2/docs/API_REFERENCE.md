# API Reference

## Module Structure

```
generative-pretrained-transformer-2.src
├── config          # Configuration classes
├── model           # GPT-2 model
├── data_loader     # Data loading utilities
├── train           # Training logic
├── evaluate        # Evaluation metrics
├── optimize        # Hyperparameter optimization
└── inference       # Text generation
```

## Configuration Classes

### GPT2Config

Model architecture configuration.

**Attributes:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| vocab_size | int | 50257 | Vocabulary size |
| context_length | int | 1024 | Maximum sequence length |
| d_model | int | 768 | Embedding dimension |
| num_heads | int | 12 | Number of attention heads |
| num_layers | int | 12 | Number of transformer blocks |
| d_ff | int | 3072 | Feed-forward dimension |
| dropout | float | 0.1 | Dropout rate |
| attention_dropout | float | 0.1 | Attention dropout rate |
| residual_dropout | float | 0.1 | Residual dropout rate |
| layer_norm_epsilon | float | 1e-5 | Layer norm epsilon |
| initializer_range | float | 0.02 | Weight initialization std |
| use_bias | bool | True | Use bias in linear layers |

**Example:**
```python
from generative-pretrained-transformer-2.src.config import GPT2Config

config = GPT2Config(
    d_model=1024,
    num_heads=16,
    num_layers=24
)
```

### TrainingConfig

Training hyperparameters.

**Attributes:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| batch_size | int | 8 | Training batch size |
| learning_rate | float | 3e-4 | Initial learning rate |
| weight_decay | float | 0.01 | L2 regularization |
| beta1 | float | 0.9 | Adam beta1 |
| beta2 | float | 0.95 | Adam beta2 |
| epsilon | float | 1e-8 | Adam epsilon |
| max_epochs | int | 10 | Maximum training epochs |
| max_training_hours | float | 12.0 | Maximum training time in hours |
| warmup_steps | int | 2000 | LR warmup steps |
| gradient_clip | float | 1.0 | Gradient clipping threshold |
| accumulation_steps | int | 1 | Gradient accumulation steps |
| save_every | int | 1000 | Checkpoint save frequency |
| eval_every | int | 500 | Validation frequency |
| log_every | int | 100 | Logging frequency |
| checkpoint_dir | str | "checkpoints" | Checkpoint directory |
| device | str | auto | Device (cuda/cpu) |
| num_workers | int | 4 | DataLoader workers |
| pin_memory | bool | True | Pin memory for GPU |
| mixed_precision | bool | True | Use AMP |
| gradient_checkpointing | bool | False | Gradient checkpointing |

### DataConfig

Dataset configuration.

**Attributes:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dataset_name | str | "wikitext" | HuggingFace dataset name |
| dataset_config | str | "wikitext-2-raw-v1" | Dataset configuration |
| train_split | str | "train" | Training split name |
| validation_split | str | "validation" | Validation split name |
| test_split | str | "test" | Test split name |
| max_length | int | 1024 | Maximum sequence length |
| cache_dir | str | None | Cache directory |
| preprocessing_num_workers | int | 4 | Preprocessing workers |

### OptunaConfig

Hyperparameter optimization configuration.

**Attributes:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_trials | int | 50 | Number of trials |
| timeout | int | None | Timeout in seconds |
| study_name | str | "gpt2_optimization" | Study name |
| storage | str | None | Database URL |
| direction | str | "minimize" | Optimization direction |
| sampler | str | "TPE" | Sampler algorithm |
| pruner | str | "MedianPruner" | Pruner algorithm |
| n_startup_trials | int | 10 | Startup trials |
| n_warmup_steps | int | 5 | Warmup steps |

### InferenceConfig

Text generation configuration.

**Attributes:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_new_tokens | int | 100 | Maximum tokens to generate |
| temperature | float | 0.8 | Sampling temperature |
| top_k | int | 50 | Top-k sampling |
| top_p | float | 0.95 | Nucleus sampling |
| repetition_penalty | float | 1.2 | Repetition penalty |
| do_sample | bool | True | Use sampling vs greedy |
| num_return_sequences | int | 1 | Number of sequences |
| stream | bool | True | Stream output |

## Model Classes

### GPT2Model

Main GPT-2 transformer model.

**Constructor:**
```python
GPT2Model(config: GPT2Config)
```

**Methods:**

#### forward
```python
forward(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

Forward pass through the model.

**Parameters:**
- `input_ids`: Token IDs, shape (batch_size, sequence_length)
- `labels`: Target labels, shape (batch_size, sequence_length)

**Returns:**
- `logits`: Predictions, shape (batch_size, sequence_length, vocab_size)
- `loss`: Cross-entropy loss (if labels provided)

**Example:**
```python
model = GPT2Model(config)
logits, loss = model(input_ids, labels)
```

#### generate
```python
generate(
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    do_sample: bool = True
) -> Generator[torch.Tensor, None, torch.Tensor]
```

Generate text autoregressively.

**Parameters:**
- `input_ids`: Starting tokens, shape (batch_size, sequence_length)
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_k`: Top-k sampling
- `top_p`: Nucleus sampling
- `repetition_penalty`: Repetition penalty
- `do_sample`: Use sampling vs greedy

**Yields:**
- Generated tokens one at a time

**Returns:**
- Complete generated sequence

**Example:**
```python
for token in model.generate(input_ids, max_new_tokens=100):
    print(tokenizer.decode(token.tolist()))
```

### MultiHeadAttention

Multi-head self-attention mechanism.

**Constructor:**
```python
MultiHeadAttention(
    d_model: int,
    num_heads: int,
    dropout: float,
    use_bias: bool
)
```

**Forward:**
```python
forward(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor = None
) -> torch.Tensor
```

### FeedForward

Position-wise feed-forward network.

**Constructor:**
```python
FeedForward(
    d_model: int,
    d_ff: int,
    dropout: float,
    use_bias: bool
)
```

**Forward:**
```python
forward(x: torch.Tensor) -> torch.Tensor
```

### DecoderBlock

Transformer decoder block.

**Constructor:**
```python
DecoderBlock(
    d_model: int,
    num_heads: int,
    d_ff: int,
    dropout: float,
    attention_dropout: float,
    layer_norm_epsilon: float,
    use_bias: bool
)
```

**Forward:**
```python
forward(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor = None
) -> torch.Tensor
```

## Training Classes

### Trainer

Model training orchestration.

**Constructor:**
```python
Trainer(
    model: GPT2Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    checkpoint_dir: Optional[str] = None
)
```

**Methods:**

#### train
```python
train() -> None
```

Execute full training loop.

**Example:**
```python
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()
```

#### train_epoch
```python
train_epoch() -> float
```

Train for one epoch.

**Returns:**
- Average training loss

#### validate
```python
validate() -> float
```

Validate on validation set.

**Returns:**
- Average validation loss

#### save_checkpoint
```python
save_checkpoint(filename: str) -> None
```

Save model checkpoint.

**Parameters:**
- `filename`: Checkpoint filename

#### load_checkpoint
```python
load_checkpoint(filepath: str) -> None
```

Load model checkpoint.

**Parameters:**
- `filepath`: Path to checkpoint

## Evaluation Classes

### Evaluator

Model evaluation.

**Constructor:**
```python
Evaluator(
    model: GPT2Model,
    test_loader: DataLoader,
    device: str = 'cuda'
)
```

**Methods:**

#### evaluate
```python
evaluate() -> Dict[str, float]
```

Evaluate model on test set.

**Returns:**
- Dictionary with metrics:
  - `loss`: Average test loss
  - `perplexity`: Test perplexity

**Example:**
```python
evaluator = Evaluator(model, test_loader)
metrics = evaluator.evaluate()
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

#### print_metrics
```python
print_metrics(metrics: Dict[str, float]) -> None
```

Print formatted metrics.

## Optimization Classes

### OptunaOptimizer

Hyperparameter optimization with Optuna.

**Constructor:**
```python
OptunaOptimizer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_config: GPT2Config,
    training_config: TrainingConfig,
    optuna_config: OptunaConfig
)
```

**Methods:**

#### optimize
```python
optimize() -> Dict[str, Any]
```

Run hyperparameter optimization.

**Returns:**
- Best hyperparameters

**Example:**
```python
optimizer = OptunaOptimizer(train_loader, val_loader, config, train_config, optuna_config)
best_params = optimizer.optimize()
```

## Inference Classes

### TextGenerator

Text generation interface.

**Constructor:**
```python
TextGenerator(
    model_path: str,
    device: str = 'cuda'
)
```

**Methods:**

#### generate_text
```python
generate_text(
    prompt: str,
    config: InferenceConfig
) -> None
```

Generate text from prompt.

**Parameters:**
- `prompt`: Input text
- `config`: Generation configuration

**Example:**
```python
generator = TextGenerator('checkpoints/best_model.pth')
config = InferenceConfig(max_new_tokens=100)
generator.generate_text("Once upon a time", config)
```

#### interactive_mode
```python
interactive_mode(config: InferenceConfig) -> None
```

Start interactive generation session.

## Data Loading Functions

### load_text_data
```python
load_text_data(
    data_config: DataConfig
) -> Tuple[Dataset, Dataset, Dataset, GPT2Tokenizer]
```

Load and preprocess text dataset.

**Returns:**
- `train_dataset`: Training dataset
- `val_dataset`: Validation dataset
- `test_dataset`: Test dataset
- `tokenizer`: GPT-2 tokenizer

### get_dataloaders
```python
get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    training_config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]
```

Create data loaders.

**Returns:**
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `test_loader`: Test data loader

## Command Line Interface

### main.py

Entry point for training, evaluation, and optimization.

**Commands:**

#### train
```bash
python -m generative-pretrained-transformer-2.src.main train [OPTIONS]
```

#### evaluate
```bash
python -m generative-pretrained-transformer-2.src.main evaluate [OPTIONS]
```

#### optimize
```bash
python -m generative-pretrained-transformer-2.src.main optimize [OPTIONS]
```

### inference.py

Entry point for text generation.

```bash
python -m generative-pretrained-transformer-2.src.inference [OPTIONS]
```

See documentation for full command options.
