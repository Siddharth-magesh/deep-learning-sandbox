# CLIP Architecture - Detailed Technical Documentation

## 🏛️ Overview

CLIP (Contrastive Language-Image Pretraining) uses a dual-encoder architecture where both images and text are mapped to the same embedding space. The model learns by maximizing the cosine similarity between correct image-text pairs while minimizing similarity with incorrect pairs.

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIP Model                            │
├─────────────────────────────┬───────────────────────────────┤
│      Vision Encoder         │       Text Encoder            │
│                             │                               │
│  ┌──────────────────┐      │   ┌──────────────────┐       │
│  │ Input Image      │      │   │ Input Text       │       │
│  │ (224×224×3)      │      │   │ (77 tokens)      │       │
│  └────────┬─────────┘      │   └────────┬─────────┘       │
│           ↓                 │            ↓                  │
│  ┌──────────────────┐      │   ┌──────────────────┐       │
│  │ Patch Embedding  │      │   │ Token Embedding  │       │
│  │ (16×16 patches)  │      │   │ + Positional     │       │
│  └────────┬─────────┘      │   └────────┬─────────┘       │
│           ↓                 │            ↓                  │
│  ┌──────────────────┐      │   ┌──────────────────┐       │
│  │ CLS Token +      │      │   │ Transformer      │       │
│  │ Pos Embedding    │      │   │ Blocks (×8)      │       │
│  └────────┬─────────┘      │   └────────┬─────────┘       │
│           ↓                 │            ↓                  │
│  ┌──────────────────┐      │   ┌──────────────────┐       │
│  │ Transformer      │      │   │ LayerNorm        │       │
│  │ Blocks (×12)     │      │   └────────┬─────────┘       │
│  └────────┬─────────┘      │            ↓                  │
│           ↓                 │   ┌──────────────────┐       │
│  ┌──────────────────┐      │   │ Projection       │       │
│  │ LayerNorm        │      │   │ (embed → 512)    │       │
│  └────────┬─────────┘      │   └────────┬─────────┘       │
│           ↓                 │            ↓                  │
│  ┌──────────────────┐      │   ┌──────────────────┐       │
│  │ Projection       │      │   │ L2 Normalize     │       │
│  │ (768 → 512)      │      │   └────────┬─────────┘       │
│  └────────┬─────────┘      │            ↓                  │
│           ↓                 │   ┌──────────────────┐       │
│  ┌──────────────────┐      │   │ Text Features    │       │
│  │ L2 Normalize     │      │   │ (batch × 512)    │       │
│  └────────┬─────────┘      │   └────────┬─────────┘       │
│           ↓                 │            │                  │
│  ┌──────────────────┐      │            │                  │
│  │ Image Features   │      │            │                  │
│  │ (batch × 512)    │      │            │                  │
│  └────────┬─────────┘      │            │                  │
│           └─────────────────┴────────────┘                  │
│                      ↓                                      │
│           ┌────────────────────┐                           │
│           │ Similarity Matrix   │                           │
│           │ (batch × batch)    │                           │
│           │ × temperature      │                           │
│           └──────────┬─────────┘                           │
│                      ↓                                      │
│           ┌────────────────────┐                           │
│           │ Contrastive Loss   │                           │
│           │ (symmetric)        │                           │
│           └────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 Component Breakdown

### 1. Vision Transformer (ViT)

#### **Input Processing**
```python
Input: (B, 3, 224, 224)  # Batch of RGB images
```

#### **Patch Embedding Layer**
Divides image into non-overlapping patches and projects to embedding dimension.

```python
# Conv2d acts as patch extraction + linear projection
patch_embed = Conv2d(
    in_channels=3,
    out_channels=768,  # embed_dim
    kernel_size=16,    # patch_size
    stride=16
)

Output: (B, 196, 768)  # 196 = (224/16)^2 patches
```

**Shape Transformations:**
1. Input: `(B, 3, 224, 224)`
2. After Conv2d: `(B, 768, 14, 14)`
3. Flatten: `(B, 768, 196)`
4. Transpose: `(B, 196, 768)`

#### **CLS Token & Positional Encoding**
```python
# CLS token: learnable classification token
cls_token = Parameter(torch.zeros(1, 1, 768))
cls_expanded = cls_token.expand(B, -1, -1)  # (B, 1, 768)

# Positional embeddings: learnable position info
pos_embed = Parameter(torch.zeros(1, 197, 768))  # 196 + 1 (CLS)

# Combine
x = concat([cls_token, patch_embeddings], dim=1)  # (B, 197, 768)
x = x + pos_embed  # Add positional information
```

#### **Transformer Blocks (×12)**

Each block contains:

**1. Multi-Head Self-Attention (MHSA)**
```python
# Pre-LayerNorm
x_norm = LayerNorm(x)

# QKV projection
qkv = Linear(768, 768*3)(x_norm)
q, k, v = split(qkv, num_heads=12)  # Each head: 64-dim

# Scaled dot-product attention
attention = softmax(q @ k.T / sqrt(64))
output = attention @ v

# Output projection
output = Linear(768, 768)(output)
x = x + Dropout(output)  # Residual connection
```

**2. Feed-Forward Network (MLP)**
```python
# Pre-LayerNorm
x_norm = LayerNorm(x)

# Two-layer MLP with GELU
hidden = Linear(768, 3072)(x_norm)  # mlp_ratio=4.0
hidden = GELU()(hidden)
hidden = Dropout(hidden)
output = Linear(3072, 768)(hidden)
output = Dropout(output)

x = x + output  # Residual connection
```

#### **Output Projection**
```python
# Extract CLS token
x = x[:, 0]  # (B, 768)

# Final normalization
x = LayerNorm(x)

# Project to embedding space
x = Linear(768, 512)(x)  # (B, 512)

# L2 normalize
x = x / ||x||₂
```

### 2. Text Transformer

#### **Input Processing**
```python
Input: (B, 77)  # Tokenized text (max 77 tokens)
```

#### **Token Embedding**
```python
# Map tokens to embeddings
token_embed = Embedding(
    num_embeddings=49408,  # vocab_size
    embedding_dim=512
)
x = token_embed(text)  # (B, 77, 512)

# Add positional embeddings
pos_embed = Parameter(torch.zeros(1, 77, 512))
x = x + pos_embed[:, :seq_len, :]
```

#### **Transformer Blocks (×8)**

Same structure as Vision Transformer but:
- Embedding dimension: 512 (vs 768)
- Number of heads: 8 (vs 12)
- Fewer layers: 8 (vs 12)

#### **Output Pooling**
```python
# Use end-of-sequence token (argmax trick)
eos_idx = text.argmax(dim=-1)  # Find EOS position
x = x[range(B), eos_idx]  # (B, 512)

# Project (already 512-dim, but adds flexibility)
x = Linear(512, 512)(x)

# L2 normalize
x = x / ||x||₂
```

### 3. Contrastive Loss

#### **Similarity Computation**
```python
# Image features: (B, 512)
# Text features: (B, 512)

# Cosine similarity matrix
logits = image_features @ text_features.T  # (B, B)

# Temperature scaling
temperature = 0.07
logits = logits / temperature
```

#### **Loss Calculation**
```python
# Ground truth: diagonal elements (correct pairs)
labels = torch.arange(B)  # [0, 1, 2, ..., B-1]

# Image-to-Text loss
loss_i2t = CrossEntropy(logits, labels)

# Text-to-Image loss
loss_t2i = CrossEntropy(logits.T, labels)

# Symmetric loss
total_loss = (loss_i2t + loss_t2i) / 2
```

**Interpretation:**
- Each image should match its corresponding text (diagonal)
- All other pairs are negatives (off-diagonal)
- Batch size determines number of negatives

## 📊 Detailed Specifications

### Vision Transformer Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input Size | 224×224×3 | RGB images |
| Patch Size | 16×16 | Non-overlapping patches |
| Num Patches | 196 | (224/16)² |
| Embed Dim | 768 | Hidden dimension |
| Depth | 12 | Transformer layers |
| Num Heads | 12 | Attention heads |
| Head Dim | 64 | 768/12 |
| MLP Ratio | 4.0 | Hidden/Embed ratio |
| Dropout | 0.1 | Dropout rate |
| Output Dim | 512 | Final projection |

### Text Transformer Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vocab Size | 49,408 | Token vocabulary |
| Max Length | 77 | Max sequence length |
| Embed Dim | 512 | Hidden dimension |
| Depth | 8 | Transformer layers |
| Num Heads | 8 | Attention heads |
| Head Dim | 64 | 512/8 |
| MLP Ratio | 4.0 | Hidden/Embed ratio |
| Dropout | 0.1 | Dropout rate |
| Output Dim | 512 | Final projection |

### Parameter Count

```python
Vision Transformer:  ~86M parameters
  - Patch embedding: ~590K
  - Transformer blocks: ~85M
  - Output projection: ~393K

Text Transformer: ~63M parameters
  - Token embedding: ~25M
  - Transformer blocks: ~37M
  - Output projection: ~262K

Total: ~149M parameters
```

## 🔄 Training Dynamics

### Forward Pass

1. **Encode Images** → Vision features (B, 512)
2. **Encode Text** → Text features (B, 512)
3. **Compute Similarity** → Logits (B, B)
4. **Apply Temperature** → Scaled logits
5. **Compute Loss** → Contrastive loss

### Backward Pass

1. Gradients flow through both encoders
2. Both vision and text models are updated
3. Temperature is learnable (optional)

### Optimization

- **Optimizer**: Adam
- **Learning Rate**: 1e-4 (default)
- **Weight Decay**: 1e-4
- **Scheduler**: Cosine Annealing
- **Batch Size**: 64 (default)

### Key Design Choices

1. **L2 Normalization**: Ensures similarity is pure cosine similarity
2. **Temperature Scaling**: Controls the "peakiness" of the distribution
3. **Symmetric Loss**: Treats image→text and text→image equally
4. **Learnable Positions**: Adapts to dataset characteristics
5. **Pre-LayerNorm**: More stable training than post-norm

## 🎯 Why This Architecture Works

1. **Shared Embedding Space**: Images and text projected to same space
2. **Large Batch Size**: More negative samples → better contrastive learning
3. **Symmetric Training**: Bidirectional understanding
4. **No Dense Supervision**: Learns from natural image-text pairs
5. **Scalable**: Works with any image-text dataset

## 📈 Computational Complexity

### Vision Transformer
- **Patch Embedding**: O(N × C × P²)
- **Self-Attention**: O(N² × D) per layer
- **Total**: O(12 × N² × D) where N=197, D=768

### Text Transformer
- **Token Embedding**: O(L × D)
- **Self-Attention**: O(L² × D) per layer
- **Total**: O(8 × L² × D) where L=77, D=512

### Memory Requirements

**GPU Memory (batch_size=64):**
- Input data: ~38 MB
- Model parameters: ~596 MB
- Activations: ~4-6 GB
- Gradients: ~596 MB
- **Total**: ~6-8 GB

## 🔧 Variations & Extensions

### Possible Modifications

1. **Larger Models**: Increase depth/width for more capacity
2. **Different Pooling**: Try attention pooling instead of CLS/EOS
3. **Data Augmentation**: Add image augmentation for robustness
4. **Hard Negatives**: Mine hard negative pairs
5. **Multi-Task**: Add auxiliary objectives

### Architecture Variants

- **CLIP-ResNet**: Use ResNet instead of ViT
- **Smaller Models**: Reduce depth for faster training
- **Larger Vocabulary**: Better text representation
- **Longer Sequences**: Support longer captions

## 📚 Implementation Notes

All components are implemented from scratch in:
- `src/vision_transformer.py` - Vision encoder
- `src/text_transformer.py` - Text encoder
- `src/modules/transformer.py` - Transformer blocks
- `src/modules/multi_head_attention.py` - Attention mechanism
- `src/modules/multi_layer_perceptron.py` - Feed-forward network
- `src/clip.py` - Main CLIP model and loss
