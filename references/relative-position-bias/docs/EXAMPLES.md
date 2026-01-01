# Usage Examples

## Basic Usage

### Creating Relative Position Bias

```python
from modules import RelativePositionBias

rpb_1d = RelativePositionBias(
    num_heads=8,
    seq_len=16,
    bias_type="1d",
    init_std=0.02
)

rpb_2d = RelativePositionBias(
    num_heads=8,
    window_size=(7, 7),
    bias_type="2d",
    init_std=0.02
)

bias_1d = rpb_1d()
bias_2d = rpb_2d()

print(f"1D Bias shape: {bias_1d.shape}")
print(f"2D Bias shape: {bias_2d.shape}")
```

### Using Multi-Head Attention

```python
import torch
from modules import MultiHeadAttention

batch_size = 4
seq_len = 49
embed_dim = 96
num_heads = 4

x = torch.randn(batch_size, seq_len, embed_dim)

attn_with_rpb = MultiHeadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    dropout=0.1,
    use_relative_position=True,
    rpb_kwargs={
        'num_heads': num_heads,
        'window_size': (7, 7),
        'bias_type': '2d',
        'init_std': 0.02
    }
)

output = attn_with_rpb(x)
print(f"Output shape: {output.shape}")
```

### Building a Vision Transformer

```python
from models import VisionTransformer

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.1,
    use_relative_position=True,
    rpb_kwargs={
        'num_heads': 12,
        'window_size': (14, 14),
        'bias_type': '2d',
        'init_std': 0.02
    }
)

images = torch.randn(8, 3, 224, 224)
logits = model(images)
print(f"Logits shape: {logits.shape}")
```

## Training Examples

### Simple Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import VisionTransformer
from data import SyntheticImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=10,
    embed_dim=192,
    depth=6,
    num_heads=6,
    use_relative_position=True,
    rpb_kwargs={'num_heads': 6, 'window_size': (14, 14), 'bias_type': '2d'}
).to(device)

train_dataset = SyntheticImageDataset(
    num_samples=1000,
    img_size=224,
    patch_size=16
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

for epoch in range(10):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

### Training with Experiments Module

```python
from experiments import train_epoch, evaluate, save_checkpoint

num_epochs = 50
best_val_acc = 0.0

for epoch in range(1, num_epochs + 1):
    train_metrics = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    val_metrics = evaluate(
        model, val_loader, criterion, device
    )
    
    print(f"Epoch {epoch}/{num_epochs}")
    print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
    print(f"  Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
    
    if val_metrics['accuracy'] > best_val_acc:
        best_val_acc = val_metrics['accuracy']
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            'checkpoints/best_model.pth'
        )
        print(f"  New best model saved!")
```

## Visualization Examples

### Visualize Relative Position Bias

```python
from modules import RelativePositionBias
from experiments import visualize_relative_position_bias, visualize_bias_table
from pathlib import Path

Path("visualizations").mkdir(exist_ok=True)

rpb = RelativePositionBias(
    num_heads=8,
    window_size=(7, 7),
    bias_type="2d",
    init_std=0.02
)

bias = rpb()

visualize_relative_position_bias(
    bias,
    save_path="visualizations/bias_heatmap.png",
    title="2D Relative Position Bias (Head 0)",
    cmap="coolwarm"
)

visualize_bias_table(
    rpb.relative_position_bias_table,
    save_path="visualizations/bias_table.png",
    title="Learned Bias Values"
)

print("Visualizations saved!")
```

### Visualize Attention Patterns

```python
import torch
from modules import MultiHeadAttention
from experiments import visualize_attention_weights

batch_size = 1
seq_len = 49
embed_dim = 96
num_heads = 4

x = torch.randn(batch_size, seq_len, embed_dim)

attn = MultiHeadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    use_relative_position=True,
    rpb_kwargs={'num_heads': num_heads, 'window_size': (7, 7), 'bias_type': '2d'}
)

attn.eval()
with torch.no_grad():
    output = attn(x)
    
    Q, K, V = attn.qkv(x).chunk(3, dim=-1)
    Q = Q.reshape(batch_size, seq_len, num_heads, -1).permute(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, -1).permute(0, 2, 1, 3)
    
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
    bias = attn.relative_position_bias()
    attn_scores = attn_scores + bias.unsqueeze(0)
    attn_weights = torch.softmax(attn_scores, dim=-1)

visualize_attention_weights(
    attn_weights,
    save_path="visualizations/attention_pattern.png",
    title="Attention Pattern with Relative Position Bias"
)
```

## Configuration Examples

### Small Model

```yaml
project:
  name: "vit_small"
  seed: 42
  device: "cuda"

input:
  batch_size: 128
  embed_dim: 384

attention:
  num_heads: 6
  dropout: 0.0

relative_position_bias:
  enabled: true
  type: "2d"
  window_size: [14, 14]
  init_std: 0.02
```

### Large Model

```yaml
project:
  name: "vit_large"
  seed: 42
  device: "cuda"

input:
  batch_size: 32
  embed_dim: 1024

attention:
  num_heads: 16
  dropout: 0.1

relative_position_bias:
  enabled: true
  type: "2d"
  window_size: [14, 14]
  init_std: 0.01
```

### Without Relative Position Bias

```yaml
relative_position_bias:
  enabled: false

absolute_position_embedding:
  enabled: true
  type: "learned"
```

## Advanced Examples

### Custom Transformer Block

```python
import torch.nn as nn
from modules import MultiHeadAttention

class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, use_rpb=True):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_relative_position=use_rpb,
            rpb_kwargs={'num_heads': num_heads, 'window_size': (7, 7), 'bias_type': '2d'}
            if use_rpb else None
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Hierarchical Model

```python
class HierarchicalViT(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stage1 = VisionTransformer(
            img_size=224,
            patch_size=4,
            embed_dim=96,
            depth=2,
            num_heads=3,
            use_relative_position=True,
            rpb_kwargs={'num_heads': 3, 'window_size': (7, 7), 'bias_type': '2d'}
        )
        
        self.stage2 = VisionTransformer(
            img_size=56,
            patch_size=2,
            embed_dim=192,
            depth=2,
            num_heads=6,
            use_relative_position=True,
            rpb_kwargs={'num_heads': 6, 'window_size': (7, 7), 'bias_type': '2d'}
        )
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x
```

### Mixed Bias Types

```python
class MixedBiasModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.attn_1d = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_relative_position=True,
            rpb_kwargs={'num_heads': num_heads, 'seq_len': 196, 'bias_type': '1d'}
        )
        
        self.attn_2d = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_relative_position=True,
            rpb_kwargs={'num_heads': num_heads, 'window_size': (14, 14), 'bias_type': '2d'}
        )
        
    def forward(self, x):
        x1 = self.attn_1d(x)
        x2 = self.attn_2d(x)
        return x1 + x2
```

## Inference Examples

### Single Image Inference

```python
from PIL import Image
import torchvision.transforms as transforms

model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("test_image.jpg")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {probabilities[0, predicted_class].item():.2%}")
```

### Batch Inference

```python
from torch.utils.data import DataLoader

model.eval()

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_predictions.extend(preds.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())

print(f"Total predictions: {len(all_predictions)}")
```

## Debugging Examples

### Check Bias Values

```python
rpb = RelativePositionBias(num_heads=4, window_size=(7, 7), bias_type="2d")

print(f"Bias table shape: {rpb.relative_position_bias_table.shape}")
print(f"Bias table mean: {rpb.relative_position_bias_table.mean():.6f}")
print(f"Bias table std: {rpb.relative_position_bias_table.std():.6f}")
print(f"Bias table min: {rpb.relative_position_bias_table.min():.6f}")
print(f"Bias table max: {rpb.relative_position_bias_table.max():.6f}")
```

### Verify Attention Output

```python
attn = MultiHeadAttention(
    embed_dim=96,
    num_heads=4,
    use_relative_position=True,
    rpb_kwargs={'num_heads': 4, 'window_size': (7, 7), 'bias_type': '2d'}
)

x = torch.randn(2, 49, 96)
output = attn(x)

assert output.shape == x.shape, "Output shape mismatch!"
assert not torch.isnan(output).any(), "NaN detected in output!"
assert not torch.isinf(output).any(), "Inf detected in output!"

print("All checks passed!")
```
