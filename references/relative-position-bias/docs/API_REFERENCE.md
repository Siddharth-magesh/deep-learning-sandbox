# API Reference

## Modules

### relative_position.py

#### `get_relative_position_index_1d(seq_len: int) -> torch.Tensor`

Compute relative position indices for 1D sequences.

**Parameters**:
- `seq_len` (int): Length of the sequence

**Returns**:
- `torch.Tensor`: Relative position index matrix of shape (seq_len, seq_len)

**Example**:
```python
indices = get_relative_position_index_1d(16)
```

---

#### `get_relative_position_index_2d(window_size: tuple[int, int]) -> torch.Tensor`

Compute relative position indices for 2D windows.

**Parameters**:
- `window_size` (tuple[int, int]): (height, width) of the window

**Returns**:
- `torch.Tensor`: Relative position index matrix of shape (H×W, H×W)

**Example**:
```python
indices = get_relative_position_index_2d((7, 7))
```

---

#### `RelativePositionBias`

Learnable relative position bias for attention mechanisms.

**Constructor Parameters**:
- `num_heads` (int): Number of attention heads
- `window_size` (tuple[int, int] | None): Window size for 2D bias
- `seq_len` (int | None): Sequence length for 1D bias
- `bias_type` (str): Type of bias, either "1d" or "2d"
- `init_std` (float): Standard deviation for bias table initialization (default: 0.02)

**Attributes**:
- `num_heads` (int): Number of attention heads
- `bias_type` (str): Type of bias being used
- `num_relative_positions` (int): Total number of unique relative positions
- `relative_position_bias_table` (nn.Parameter): Learnable bias table
- `relative_position_index` (torch.Tensor): Pre-computed position indices (buffer)

**Methods**:

##### `forward() -> torch.Tensor`

Generate relative position bias matrix.

**Returns**:
- `torch.Tensor`: Bias matrix of shape (num_heads, seq_len, seq_len)

**Example**:
```python
rpb = RelativePositionBias(
    num_heads=8,
    window_size=(7, 7),
    bias_type="2d",
    init_std=0.02
)
bias = rpb()
```

---

### attention.py

#### `ScaledDotProductAttention`

Scaled dot-product attention with optional relative position bias.

**Constructor Parameters**:
- `scale` (bool): Whether to scale attention by 1/√d_k (default: True)
- `dropout` (float): Dropout rate (default: 0.0)

**Methods**:

##### `forward(query, key, value, relative_position_bias=None) -> tuple[torch.Tensor, torch.Tensor]`

Compute attention output.

**Parameters**:
- `query` (torch.Tensor): Query tensor of shape (B, H, N, D)
- `key` (torch.Tensor): Key tensor of shape (B, H, N, D)
- `value` (torch.Tensor): Value tensor of shape (B, H, N, D)
- `relative_position_bias` (torch.Tensor | None): Bias of shape (H, N, N)

**Returns**:
- `output` (torch.Tensor): Attention output of shape (B, H, N, D)
- `attn` (torch.Tensor): Attention weights of shape (B, H, N, N)

---

#### `MultiHeadAttention`

Multi-head self-attention with optional relative position bias.

**Constructor Parameters**:
- `embed_dim` (int): Total dimension of the model
- `num_heads` (int): Number of attention heads
- `dropout` (float): Dropout rate (default: 0.0)
- `use_relative_position` (bool): Whether to use relative position bias (default: False)
- `rpb_kwargs` (dict | None): Keyword arguments for RelativePositionBias

**Attributes**:
- `embed_dim` (int): Embedding dimension
- `num_heads` (int): Number of attention heads
- `head_dim` (int): Dimension per head (embed_dim // num_heads)
- `qkv` (nn.Linear): Combined Q, K, V projection layer
- `attention` (ScaledDotProductAttention): Attention module
- `relative_position_bias` (RelativePositionBias | None): RPB module if enabled

**Methods**:

##### `forward(x: torch.Tensor) -> torch.Tensor`

Apply multi-head attention.

**Parameters**:
- `x` (torch.Tensor): Input tensor of shape (B, N, C)

**Returns**:
- `torch.Tensor`: Output tensor of shape (B, N, C)

**Example**:
```python
attn = MultiHeadAttention(
    embed_dim=96,
    num_heads=4,
    use_relative_position=True,
    rpb_kwargs={
        'num_heads': 4,
        'window_size': (7, 7),
        'bias_type': '2d'
    }
)
output = attn(x)
```

---

### models/transformer.py

#### `TransformerBlock`

Standard transformer block with attention and MLP.

**Constructor Parameters**:
- `embed_dim` (int): Embedding dimension
- `num_heads` (int): Number of attention heads
- `mlp_ratio` (float): Ratio of MLP hidden dim to embedding dim (default: 4.0)
- `dropout` (float): Dropout rate (default: 0.0)
- `use_relative_position` (bool): Whether to use relative position bias
- `rpb_kwargs` (dict | None): Arguments for RelativePositionBias

**Methods**:

##### `forward(x: torch.Tensor) -> torch.Tensor`

Process input through transformer block.

**Parameters**:
- `x` (torch.Tensor): Input of shape (B, N, C)

**Returns**:
- `torch.Tensor`: Output of shape (B, N, C)

---

#### `VisionTransformer`

Complete Vision Transformer model.

**Constructor Parameters**:
- `img_size` (int): Input image size
- `patch_size` (int): Size of image patches
- `in_channels` (int): Number of input channels
- `num_classes` (int): Number of output classes
- `embed_dim` (int): Embedding dimension
- `depth` (int): Number of transformer blocks
- `num_heads` (int): Number of attention heads
- `mlp_ratio` (float): MLP hidden dimension ratio (default: 4.0)
- `dropout` (float): Dropout rate (default: 0.0)
- `use_relative_position` (bool): Use relative position bias
- `rpb_kwargs` (dict | None): Arguments for RelativePositionBias

**Methods**:

##### `forward(x: torch.Tensor) -> torch.Tensor`

Forward pass through the model.

**Parameters**:
- `x` (torch.Tensor): Input images of shape (B, C, H, W)

**Returns**:
- `torch.Tensor`: Class logits of shape (B, num_classes)

**Example**:
```python
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    use_relative_position=True,
    rpb_kwargs={
        'num_heads': 12,
        'window_size': (14, 14),
        'bias_type': '2d'
    }
)
output = model(images)
```

---

### data/dataset.py

#### `SyntheticImageDataset`

Synthetic dataset for testing and demonstrations.

**Constructor Parameters**:
- `num_samples` (int): Number of samples in dataset
- `img_size` (int): Size of square images
- `patch_size` (int): Size of patches
- `in_channels` (int): Number of input channels (default: 3)

**Methods**:

##### `__getitem__(idx: int) -> tuple[torch.Tensor, torch.Tensor]`

Get a single sample.

**Returns**:
- `image` (torch.Tensor): Random image of shape (C, H, W)
- `label` (torch.Tensor): Random label (scalar)

---

#### `PatchEmbedding`

Convert images to patch embeddings.

**Constructor Parameters**:
- `img_size` (int): Input image size
- `patch_size` (int): Size of patches
- `in_channels` (int): Number of input channels
- `embed_dim` (int): Embedding dimension

**Methods**:

##### `forward(x: torch.Tensor) -> torch.Tensor`

Convert image to patch embeddings.

**Parameters**:
- `x` (torch.Tensor): Images of shape (B, C, H, W)

**Returns**:
- `torch.Tensor`: Patch embeddings of shape (B, num_patches, embed_dim)

---

## Experiments

### experiments/train.py

#### `train_epoch(model, dataloader, criterion, optimizer, device, epoch) -> dict`

Train for one epoch.

**Parameters**:
- `model` (nn.Module): Model to train
- `dataloader` (DataLoader): Training data loader
- `criterion` (nn.Module): Loss function
- `optimizer` (torch.optim.Optimizer): Optimizer
- `device` (torch.device): Device to train on
- `epoch` (int): Current epoch number

**Returns**:
- `dict`: Metrics dictionary with 'loss' and 'accuracy'

---

#### `evaluate(model, dataloader, criterion, device) -> dict`

Evaluate model.

**Parameters**:
- `model` (nn.Module): Model to evaluate
- `dataloader` (DataLoader): Validation data loader
- `criterion` (nn.Module): Loss function
- `device` (torch.device): Device to evaluate on

**Returns**:
- `dict`: Metrics dictionary with 'loss' and 'accuracy'

---

#### `save_checkpoint(model, optimizer, epoch, metrics, save_path) -> None`

Save model checkpoint.

**Parameters**:
- `model` (nn.Module): Model to save
- `optimizer` (torch.optim.Optimizer): Optimizer state
- `epoch` (int): Current epoch
- `metrics` (dict): Training metrics
- `save_path` (str | Path): Path to save checkpoint

---

#### `load_checkpoint(model, optimizer, checkpoint_path) -> tuple[int, dict]`

Load model checkpoint.

**Parameters**:
- `model` (nn.Module): Model to load into
- `optimizer` (torch.optim.Optimizer): Optimizer to load into
- `checkpoint_path` (str | Path): Path to checkpoint file

**Returns**:
- `epoch` (int): Saved epoch number
- `metrics` (dict): Saved metrics

---

### experiments/visualize.py

#### `visualize_attention_weights(attn_weights, save_path=None, title="Attention Weights", cmap="viridis") -> None`

Visualize attention weight matrix.

**Parameters**:
- `attn_weights` (torch.Tensor): Attention weights
- `save_path` (str | Path | None): Path to save figure
- `title` (str): Plot title
- `cmap` (str): Colormap name

---

#### `visualize_relative_position_bias(bias, save_path=None, title="Relative Position Bias", cmap="coolwarm") -> None`

Visualize relative position bias matrix.

**Parameters**:
- `bias` (torch.Tensor): Bias matrix
- `save_path` (str | Path | None): Path to save figure
- `title` (str): Plot title
- `cmap` (str): Colormap name

---

#### `visualize_bias_table(bias_table, save_path=None, title="Bias Table") -> None`

Visualize bias table for all heads.

**Parameters**:
- `bias_table` (torch.Tensor): Bias table of shape (num_positions, num_heads)
- `save_path` (str | Path | None): Path to save figure
- `title` (str): Plot title

---

### experiments/config_utils.py

#### `load_config(config_path: str | Path) -> dict`

Load configuration from YAML file.

**Parameters**:
- `config_path` (str | Path): Path to config file

**Returns**:
- `dict`: Configuration dictionary

---

#### `save_config(config: dict, save_path: str | Path) -> None`

Save configuration to YAML file.

**Parameters**:
- `config` (dict): Configuration to save
- `save_path` (str | Path): Path to save file
