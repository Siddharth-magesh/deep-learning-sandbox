# API Reference - CLIP Implementation

Complete reference for all modules, classes, and functions in the CLIP implementation.

## 📦 Module Overview

```python
src/
├── clip.py                    # Main CLIP model
├── config.py                  # Configuration
├── data_loader.py             # Dataset handling
├── train.py                   # Trainer class
├── main.py                    # Entry point
├── optimize.py                # Hyperparameter optimization
├── vision_transformer.py      # Vision encoder
├── text_transformer.py        # Text encoder
└── modules/
    ├── transformer.py         # Transformer block
    ├── multi_head_attention.py
    ├── multi_layer_perceptron.py
    └── patch_embedding.py
```

---

## 🎯 Core Components

### `clip.py`

#### `CLIP`

Main CLIP model combining vision and text encoders.

```python
class CLIP(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        vision_embed_dim=768,
        vision_depth=12,
        vision_heads=12,
        vocab_size=49408,
        text_embed_dim=512,
        max_len=77,
        text_heads=8,
        text_depth=8,
        output_dim=512,
        temperature=0.07,
        vision_dropout=0.1,
        text_dropout=0.1
    )
```

**Parameters:**
- `img_size` (int): Input image size (default: 224)
- `patch_size` (int): Vision transformer patch size (default: 16)
- `vision_embed_dim` (int): Vision embedding dimension (default: 768)
- `vision_depth` (int): Number of vision transformer layers (default: 12)
- `vision_heads` (int): Number of attention heads in vision (default: 12)
- `vocab_size` (int): Text vocabulary size (default: 49408)
- `text_embed_dim` (int): Text embedding dimension (default: 512)
- `max_len` (int): Maximum text sequence length (default: 77)
- `text_heads` (int): Number of attention heads in text (default: 8)
- `text_depth` (int): Number of text transformer layers (default: 8)
- `output_dim` (int): Output embedding dimension (default: 512)
- `temperature` (float): Temperature for contrastive loss (default: 0.07)
- `vision_dropout` (float): Dropout rate for vision encoder (default: 0.1)
- `text_dropout` (float): Dropout rate for text encoder (default: 0.1)

**Methods:**

```python
def encode_image(self, image: torch.Tensor) -> torch.Tensor:
    """
    Encode images to feature vectors.
    
    Args:
        image: Input images (B, 3, H, W)
    
    Returns:
        Normalized image features (B, output_dim)
    """
```

```python
def encode_text(self, text: torch.Tensor) -> torch.Tensor:
    """
    Encode text to feature vectors.
    
    Args:
        text: Tokenized text (B, max_len)
    
    Returns:
        Normalized text features (B, output_dim)
    """
```

```python
def forward(
    self, 
    image: torch.Tensor, 
    text: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass through CLIP model.
    
    Args:
        image: Input images (B, 3, H, W)
        text: Tokenized text (B, max_len)
    
    Returns:
        logits: Similarity matrix (B, B)
        image_features: Image embeddings (B, output_dim)
        text_features: Text embeddings (B, output_dim)
    """
```

**Example:**

```python
from clip import CLIP

model = CLIP(
    vision_embed_dim=768,
    text_embed_dim=512,
    output_dim=512
)

images = torch.randn(32, 3, 224, 224)
texts = torch.randint(0, 49408, (32, 77))

logits, img_feats, txt_feats = model(images, texts)
print(logits.shape)  # (32, 32)
```

#### `CLIPLoss`

Contrastive loss function for CLIP.

```python
class CLIPLoss(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric contrastive loss.
        
        Args:
            logits: Similarity matrix (B, B)
        
        Returns:
            Scalar loss value
        """
```

**Example:**

```python
from clip import CLIPLoss

loss_fn = CLIPLoss()
logits = torch.randn(64, 64)  # Similarity matrix
loss = loss_fn(logits)
```

---

### `config.py`

#### `Config`

Configuration dataclass for all hyperparameters.

```python
@dataclass
class Config:
    # Vision Transformer
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    output_dim: int = 512
    vision_dropout: float = 0.1
    
    # Text Transformer
    vocab_size: int = 49408
    text_embed_dim: int = 512
    max_len: int = 77
    text_num_heads: int = 8
    text_depth: int = 8
    text_mlp_ratio: float = 4.0
    text_dropout: float = 0.1
    text_output_dim: int = 512
    
    # Training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    num_workers: int = 4
    pin_memory: bool = True
    save_dir: str = "./checkpoints"
```

**Methods:**

```python
def display(self) -> None:
    """Print all configuration values."""
```

**Example:**

```python
from config import Config

config = Config()
config.batch_size = 128
config.learning_rate = 5e-4
config.display()
```

---

### `data_loader.py`

#### `simple_tokenizer`

Simple hash-based tokenizer for text.

```python
def simple_tokenizer(text: str, max_length: int = 77) -> torch.Tensor:
    """
    Tokenize text using hash-based encoding.
    
    Args:
        text: Input text string
        max_length: Maximum sequence length
    
    Returns:
        Token tensor (max_length,)
    """
```

**Example:**

```python
from data_loader import simple_tokenizer

text = "A dog playing in the park"
tokens = simple_tokenizer(text, max_length=77)
print(tokens.shape)  # (77,)
```

#### `Flickr30kDataset`

PyTorch Dataset for Flickr30k.

```python
class Flickr30kDataset(Dataset):
    def __init__(
        self,
        transform=None,
        max_length=77,
        max_samples=None
    )
```

**Parameters:**
- `transform`: Image transformations (torchvision.transforms)
- `max_length` (int): Maximum text length (default: 77)
- `max_samples` (int): Limit dataset size (default: None)

**Methods:**

```python
def __len__(self) -> int:
    """Return dataset size."""

def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get image-text pair.
    
    Returns:
        image: Transformed image (3, 224, 224)
        text: Tokenized caption (max_length,)
    """
```

#### `get_data_loader`

Create DataLoader for Flickr30k.

```python
def get_data_loader(
    batch_size: int = 32,
    num_workers: int = 2,
    max_samples: int = None
) -> DataLoader:
    """
    Create data loader with standard transforms.
    
    Args:
        batch_size: Batch size
        num_workers: Number of parallel workers
        max_samples: Limit dataset size
    
    Returns:
        DataLoader instance
    """
```

**Example:**

```python
from data_loader import get_data_loader

loader = get_data_loader(batch_size=64, num_workers=4)
for images, captions in loader:
    print(images.shape, captions.shape)
    # (64, 3, 224, 224), (64, 77)
    break
```

---

### `train.py`

#### `Trainer`

Training manager for CLIP model.

```python
class Trainer:
    def __init__(self, config: Config)
```

**Attributes:**
- `config`: Configuration object
- `device`: Training device (CPU/GPU)
- `model`: CLIP model
- `loss_fn`: Loss function
- `optimizer`: Optimizer
- `scheduler`: Learning rate scheduler
- `train_loader`: DataLoader
- `best_loss`: Best loss achieved
- `checkpoint_dir`: Checkpoint save directory

**Methods:**

```python
def load_data(self, max_samples: int = None) -> None:
    """
    Load Flickr30k dataset.
    
    Args:
        max_samples: Limit dataset size
    """
```

```python
def build_model(self) -> None:
    """Initialize model, optimizer, and scheduler."""
```

```python
def train_epoch(self) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
```

```python
def save_checkpoint(self, epoch: int, avg_loss: float) -> None:
    """
    Save model checkpoint.
    
    Args:
        epoch: Current epoch number
        avg_loss: Average epoch loss
    """
```

```python
def train(self, epochs: int = None) -> nn.Module:
    """
    Main training loop.
    
    Args:
        epochs: Number of epochs (overrides config)
    
    Returns:
        Trained model
    """
```

```python
def load_checkpoint(self, checkpoint_path: str) -> int:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Epoch number from checkpoint
    """
```

**Example:**

```python
from config import Config
from train import Trainer

config = Config()
trainer = Trainer(config)
trainer.load_data(max_samples=1000)
trainer.build_model()
model = trainer.train()
```

---

### `vision_transformer.py`

#### `VisionTransformer`

Vision Transformer encoder.

```python
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 512
    )
```

**Parameters:**
- `img_size` (int): Input image size
- `patch_size` (int): Patch size for tokenization
- `in_channels` (int): Number of input channels (3 for RGB)
- `embed_dim` (int): Embedding dimension
- `depth` (int): Number of transformer blocks
- `num_heads` (int): Number of attention heads
- `mlp_ratio` (float): MLP hidden dimension ratio
- `dropout` (float): Dropout rate
- `output_dim` (int): Output projection dimension

**Methods:**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass.
    
    Args:
        x: Input images (B, C, H, W)
    
    Returns:
        Image features (B, output_dim)
    """
```

**Example:**

```python
from vision_transformer import VisionTransformer

vit = VisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12
)

images = torch.randn(32, 3, 224, 224)
features = vit(images)
print(features.shape)  # (32, 512)
```

---

### `text_transformer.py`

#### `TextTransformer`

Text Transformer encoder.

```python
class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        max_len: int = 77,
        num_heads: int = 8,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 512
    )
```

**Parameters:**
- `vocab_size` (int): Vocabulary size
- `embed_dim` (int): Embedding dimension
- `max_len` (int): Maximum sequence length
- `num_heads` (int): Number of attention heads
- `depth` (int): Number of transformer blocks
- `mlp_ratio` (float): MLP hidden dimension ratio
- `dropout` (float): Dropout rate
- `output_dim` (int): Output projection dimension

**Methods:**

```python
def forward(self, text: torch.Tensor) -> torch.Tensor:
    """
    Forward pass.
    
    Args:
        text: Tokenized text (B, L)
    
    Returns:
        Text features (B, output_dim)
    """
```

**Example:**

```python
from text_transformer import TextTransformer

text_enc = TextTransformer(
    vocab_size=49408,
    embed_dim=512,
    depth=8
)

tokens = torch.randint(0, 49408, (32, 77))
features = text_enc(tokens)
print(features.shape)  # (32, 512)
```

---

## 🧩 Module Components

### `modules/transformer.py`

#### `TransformerBlock`

Single transformer block with attention and MLP.

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    )
```

### `modules/multi_head_attention.py`

#### `MultiHeadAttention`

Multi-head self-attention mechanism.

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    )
```

### `modules/multi_layer_perceptron.py`

#### `MLP`

Feed-forward network with GELU activation.

```python
class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    )
```

### `modules/patch_embedding.py`

#### `PatchEmbedding`

Convert images to patch embeddings.

```python
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    )
```

---

## 🔧 Utility Functions

### `main.py`

```python
def main():
    """Main training entry point."""
```

### `optimize.py`

```python
def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Negative best loss (to maximize)
    """

def main():
    """Run hyperparameter optimization."""
```

---

## 📊 Type Definitions

### Common Types

```python
# Tensors
ImageTensor = torch.Tensor  # (B, 3, H, W)
TextTensor = torch.Tensor   # (B, L)
FeatureTensor = torch.Tensor  # (B, D)
LogitTensor = torch.Tensor  # (B, B)

# Data types
ImageTextPair = Tuple[torch.Tensor, torch.Tensor]
ModelOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

---

## 🎯 Usage Examples

### Complete Training Pipeline

```python
from config import Config
from train import Trainer

# 1. Configure
config = Config()
config.batch_size = 64
config.num_epochs = 20

# 2. Initialize trainer
trainer = Trainer(config)

# 3. Load data
trainer.load_data()

# 4. Build model
trainer.build_model()

# 5. Train
model = trainer.train()

# 6. Save final model
torch.save(model.state_dict(), 'final_model.pth')
```

### Inference

```python
from clip import CLIP
from data_loader import simple_tokenizer
from torchvision import transforms
from PIL import Image

# Load model
model = CLIP()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = transform(Image.open('image.jpg')).unsqueeze(0)

# Prepare text
text = simple_tokenizer("A photo of a dog").unsqueeze(0)

# Compute similarity
with torch.no_grad():
    logits, img_feat, txt_feat = model(image, text)
    similarity = logits[0, 0].item()
    print(f"Similarity: {similarity:.4f}")
```

---

## 📝 Notes

- All tensors use PyTorch convention (batch_first=True)
- Default dtype is float32
- GPU support is automatic if CUDA available
- All models support both training and inference modes
- Checkpoints include full training state for resuming
