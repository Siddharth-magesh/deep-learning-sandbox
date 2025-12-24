---
license: mit
language:
- en
tags:
- clip
- vision-language
- contrastive-learning
- image-text-matching
- pytorch
- vision-transformer
- zero-shot
- multimodal
- feature-extraction
library_name: pytorch
datasets:
- flickr30k
metrics:
- loss
pipeline_tag: feature-extraction
---

# CLIP-Flickr30k: Contrastive Language-Image Pretraining Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

This repository contains PyTorch model weights for a CLIP (Contrastive Language-Image Pretraining) implementation trained from scratch on the Flickr30k dataset.

## Model Overview

This is a **custom PyTorch implementation** of CLIP, not compatible with Hugging Face Transformers. The model learns to align images and text in a shared embedding space using contrastive learning.

### Architecture

- **Vision Encoder**: Vision Transformer (ViT)
  - Embedding dimension: 768
  - Depth: 12 layers
  - Attention heads: 12
  - Patch size: 16×16
  - Input size: 224×224

- **Text Encoder**: Transformer
  - Embedding dimension: 512
  - Depth: 8 layers
  - Attention heads: 8
  - Max sequence length: 77 tokens
  - Vocabulary size: 49,408

- **Output**: 512-dimensional embeddings (both image and text)

### Training Details

- **Dataset**: Flickr30k (1,000 image-caption pairs, 200 unique images)
- **Epochs**: 50
- **Batch Size**: 64
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Temperature**: 0.07
- **Device**: CUDA (GPU)
- **Training Time**: 8.12 hours

## Performance

| Metric | Value |
|--------|-------|
| Best Loss | **0.2570** (epoch 44) |
| Initial Loss | 4.3295 |
| Loss Reduction | 93.8% |
| Convergence | Epoch 35-40 |

### Training Progress

```
Epoch  1: Loss = 4.3295
Epoch 10: Loss = 3.3269
Epoch 20: Loss = 0.7544
Epoch 30: Loss = 0.3712
Epoch 44: Loss = 0.2570 (Best)
Epoch 50: Loss = 0.2683
```

## Model Files

This repository contains:

- `best_model.pth` - Best performing checkpoint (epoch 44, loss: 0.2570) - **598 MB**
- Additional epoch checkpoints (epochs 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)

**Note:** These are raw PyTorch state dictionaries, not Hugging Face Transformers models.

## Usage

### Installation

```bash
pip install torch torchvision pandas numpy pillow
```

### Model Architecture Code

You need to implement the model architecture to load these weights. Here's the required structure:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # Vision Transformer
        self.visual = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            output_dim=512
        )
        # Text Transformer
        self.text = TextTransformer(
            vocab_size=49408,
            embed_dim=512,
            max_len=77,
            num_heads=8,
            depth=8,
            output_dim=512
        )
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_image(self, image):
        image_features = self.visual(image)
        return F.normalize(image_features, dim=-1)
    
    def encode_text(self, text):
        text_features = self.text(text)
        return F.normalize(text_features, dim=-1)
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        logits = image_features @ text_features.T * torch.exp(self.temperature)
        return logits, image_features, text_features
```

### Loading the Model

```python
from huggingface_hub import hf_hub_download
import torch

# Download the best model checkpoint
model_path = hf_hub_download(
    repo_id="siddharth-magesh/clip-flickr30k",
    filename="best_model.pth"
)

# Initialize your model (requires architecture implementation)
model = CLIP()

# Load weights
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully!")
```

### Inference Example

```python
import torch
from torchvision import transforms
from PIL import Image

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open('your_image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Simple tokenizer (hash-based)
def tokenize(text, max_length=77):
    import numpy as np
    tokens = text.lower().split()
    idxs = [min(hash(w) % 49408, 49407) for w in tokens][:max_length]
    arr = np.zeros(max_length, dtype=np.int64)
    arr[:len(idxs)] = idxs
    return torch.tensor(arr, dtype=torch.long)

# Tokenize text
text = "a photo of a dog"
text_tensor = tokenize(text).unsqueeze(0)

# Inference
with torch.no_grad():
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tensor)
    
    # Compute similarity
    similarity = (image_features @ text_features.T).item()
    print(f"Similarity: {similarity:.4f}")
```

### Zero-Shot Image Classification

```python
def zero_shot_classification(image, texts, model):
    """
    Classify an image using text descriptions.
    
    Args:
        image: PIL Image
        texts: List of text descriptions
        model: CLIP model
    
    Returns:
        Probabilities for each text
    """
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)
    
    # Tokenize all texts
    text_tensors = torch.stack([tokenize(text) for text in texts])
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensors)
        
        # Compute similarities
        similarities = image_features @ text_features.T
        probs = F.softmax(similarities / 0.07, dim=-1)
    
    return probs[0].numpy()

# Example usage
texts = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a bird"
]
probs = zero_shot_classification(image, texts, model)

for text, prob in zip(texts, probs):
    print(f"{text}: {prob:.2%}")
```

### Image-Text Retrieval

```python
def retrieve_images(query_text, image_paths, model, top_k=5):
    """
    Retrieve most relevant images for a text query.
    
    Args:
        query_text: Text query
        image_paths: List of image file paths
        model: CLIP model
        top_k: Number of results to return
    
    Returns:
        List of (image_path, similarity) tuples
    """
    # Encode query
    query_tensor = tokenize(query_text).unsqueeze(0)
    with torch.no_grad():
        query_features = model.encode_text(query_tensor)
    
    # Encode images
    similarities = []
    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            sim = (query_features @ image_features.T).item()
        
        similarities.append((img_path, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
```

## Full Implementation

For the complete implementation including all architecture components, visit:
- **GitHub Repository**: [Include your GitHub link here]
- **Documentation**: Comprehensive docs available in the repository

Required files for full implementation:
- `clip.py` - Main CLIP model
- `vision_transformer.py` - Vision encoder
- `text_transformer.py` - Text encoder
- `modules/transformer.py` - Transformer blocks
- `modules/multi_head_attention.py` - Attention mechanism
- `modules/multi_layer_perceptron.py` - MLP layers
- `modules/patch_embedding.py` - Patch embedding

## Important Notes

1. **Not Hugging Face Transformers Compatible**: This model uses custom PyTorch code, not the Transformers library.

2. **Architecture Required**: You must implement the model architecture (see structure above) to use these weights.

3. **Simple Tokenizer**: Uses hash-based tokenization (not WordPiece or BPE).

4. **Limited Dataset**: Trained on only 1,000 image-caption pairs. For production use, retrain on the full Flickr30k dataset (158,925 pairs) or larger datasets like COCO.

5. **GPU Recommended**: Inference is faster on GPU, but CPU works fine.

## 🔧 Model Configuration

```python
config = {
    # Vision Transformer
    'img_size': 224,
    'patch_size': 16,
    'vision_embed_dim': 768,
    'vision_depth': 12,
    'vision_heads': 12,
    'vision_dropout': 0.1,
    
    # Text Transformer
    'vocab_size': 49408,
    'text_embed_dim': 512,
    'max_len': 77,
    'text_heads': 8,
    'text_depth': 8,
    'text_dropout': 0.1,
    
    # Common
    'output_dim': 512,
    'temperature': 0.07,
}
```

## Training Details

### Loss Function

Symmetric contrastive loss:
```python
loss = (cross_entropy(image_to_text_logits, labels) + 
        cross_entropy(text_to_image_logits, labels)) / 2
```

### Data Augmentation

Standard ImageNet normalization:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

### Hardware

- GPU: CUDA-enabled GPU
- Training time: ~580 seconds per epoch
- Total training: 8.12 hours (50 epochs)

## Citation

If you use this model, please cite:

```bibtex
@misc{clip-flickr30k-2025,
  author = {Siddharth Magesh},
  title = {CLIP-Flickr30k: PyTorch Implementation},
  year = {2025},
  publisher = {HuggingFace Hub},
  url = {https://huggingface.co/siddharth-magesh/clip-flickr30k}
}
```

Original CLIP paper:
```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```

## License

MIT License - See LICENSE file for details.

## Links

- **Model Card**: [Hugging Face Model Hub](https://huggingface.co/siddharth-magesh/clip-flickr30k)
- **Dataset**: [Flickr30k on Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
- **Original CLIP Paper**: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

## Contact

For questions or issues:
- Create an issue on GitHub
- Discussion tab on Hugging Face

## Acknowledgments

- OpenAI for the original CLIP architecture
- Flickr30k dataset creators
- PyTorch team

---

**Note**: This is an educational implementation. For production use, consider:
1. Training on larger datasets (COCO, Conceptual Captions, LAION)
2. Using proper tokenizers (BPE, WordPiece)
3. Pre-training on web-scale data
4. Fine-tuning for specific tasks
