---
license: mit
language:
  - en
library_name: pytorch
tags:
  - signature-verification
  - biometric-authentication
  - siamese-network
  - metric-learning
  - triplet-loss
  - custom-architecture
  - cnn
datasets:
  - siddharthmagesh/signature-verfication
metrics:
  - accuracy
  - precision
  - recall
  - f1
  - auc-roc
pipeline_tag: image-classification
---

# Siamese Network for Signature Verification

A deep learning model built entirely from scratch using PyTorch to verify the authenticity of handwritten signatures. The model uses **Siamese Networks** with **Triplet Loss** for metric learning, creating embeddings where genuine signature pairs are close together and forged signatures are far apart.

## Model Details

### Custom Architecture Overview
This model was **built from scratch** without using pre-trained weights. It consists of two main components:

#### 1. SimpleEmbeddingNetwork
A custom CNN-based feature extractor designed specifically for signature analysis:

**Architecture Layers:**
- **Conv Block 1:** 3 → 32 channels (5×5 kernel, MaxPool 2×2)
- **Conv Block 2:** 32 → 64 channels (5×5 kernel, MaxPool 2×2)
- **Conv Block 3:** 64 → 128 channels (3×3 kernel, MaxPool 2×2)
- **Conv Block 4:** 128 → 256 channels (3×3 kernel, MaxPool 2×2)

**Fully Connected Layers:**
- Linear(flattened_features, 512) + BatchNorm + ReLU + Dropout(0.5)
- Linear(512, 256) + BatchNorm + ReLU + Dropout(0.3)
- Linear(256, embedding_dim=256)

**Output:** L2-normalized embeddings (256-dimensional)

#### 2. SiameseNetwork
Wraps the embedding network with shared weights:
- **Triplet Mode:** Processes anchor, positive, and negative samples
- **Pair Mode:** Processes two images for comparison
- **Distance Computation:** Euclidean and cosine distance metrics
- **Similarity Prediction:** Built-in threshold-based classification

### Key Features
- **Custom CNN Architecture:** Optimized for signature images
- **Metric Learning:** Trained using triplet loss
- **L2 Normalization:** Embeddings are normalized for consistent distance metrics
- **Batch Normalization & Dropout:** Prevents overfitting
- **Flexible Input:** Supports both triplet and pair-wise comparisons

### Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 0.6773 |
| **Test Accuracy** | 0.6673 |
| **Precision** | 0.6840 |
| **Recall** | 0.6218 |
| **F1 Score** | 0.6514 |
| **AUC-ROC** | 0.5042 |
| **Optimal Threshold** | 0.8485 |

### Distance Statistics
- **Genuine Pair Distance:** 0.7386 ± 0.4836
- **Forged Pair Distance:** 1.1545 ± 0.4977
- **Class Separation:** 0.4160

## Dataset

**Source:** [siddharthmagesh/signature-verfication](https://www.kaggle.com/datasets/siddharthmagesh/signature-verfication) (Kaggle)

**Structure:**
- Real signatures per user
- Forged signatures per user
- Triplet dataset generation: 100 triplets per user
- Train/Val Split: 80/20

**Preprocessing:**
- Image Size: 224 × 224
- Normalization Mean: [0.861, 0.861, 0.861]
- Normalization Std: [0.274, 0.274, 0.274]
- Data Augmentation: Random affine transforms, perspective distortion

## Training Configuration

### Hyperparameters
- **Batch Size:** 32
- **Learning Rate:** 0.000973 (optimized via Optuna)
- **Weight Decay:** 0.000177
- **Triplet Margin:** 0.6836
- **Epochs:** 50
- **Scheduler Gamma:** 0.3953

### Training Details
- **Total Training Time:** 445.15 minutes (~7.4 hours)
- **Best Epoch:** 32
- **Optimizer:** Adam
- **Scheduler:** StepLR (step_size=5)

## Use Cases

✍️ **Signature Verification** - Authenticate handwritten signatures on documents  
🔐 **Biometric Authentication** - Secure access control systems  
📊 **Forensic Analysis** - Support document authentication investigations  
🖊️ **Check Fraud Detection** - Verify signatures on banking documents  

## Usage

### Simple Inference Example

```python
import torch
from PIL import Image
from torchvision import transforms
from modules.embedding_network import SimpleEmbeddingNetwork
from siamese_network import SiameseNetwork

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
embedding_net = SimpleEmbeddingNetwork(embedding_dim=256, input_size=(224, 224))
model = SiameseNetwork(embedding_net)
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.861, 0.861, 0.861], [0.274, 0.274, 0.274])
])

# Load and process images
img1 = transform(Image.open('sig1.jpg')).unsqueeze(0).to(device)
img2 = transform(Image.open('sig2.jpg')).unsqueeze(0).to(device)

# Get embeddings and compare
with torch.no_grad():
    z1, z2 = model(img1, img2, triplet_bool=False)
    distance = model.compute_distance(z1, z2).item()
    is_match = distance < 0.8485  # Optimal threshold

print(f"Distance: {distance:.4f}")
print(f"Match: {is_match}")
```

### Download Checkpoint from HuggingFace

```python
from huggingface_hub import hf_hub_download

# Download model weights
weights_path = hf_hub_download(
    repo_id="siddharth-magesh/siamese-signature-verification",
    filename="best_model.pth"
)

checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint)
```

### Compare Two Signatures

```python
# Load and preprocess images
img1 = transform(Image.open('signature1.jpg')).unsqueeze(0).to(device)
img2 = transform(Image.open('signature2.jpg')).unsqueeze(0).to(device)

# Predict similarity
with torch.no_grad():
    is_match = siamese_model.predict_similarity(
        img1, img2, 
        threshold=0.8485  # Optimal threshold from training
    )
    
print(f"Signatures match: {is_match}")

# Get embeddings and distance
with torch.no_grad():
    emb1 = siamese_model.get_embedding(img1)
    emb2 = siamese_model.get_embedding(img2)
    distance = siamese_model.compute_distance(emb1, emb2).item()
    
print(f"Embedding distance: {distance:.4f}")
```

### Batch Processing

```python
# Process multiple image pairs
image_paths = [
    ('sig1.jpg', 'sig2.jpg'),
    ('sig3.jpg', 'sig4.jpg'),
]

with torch.no_grad():
    for path1, path2 in image_paths:
        img1 = transform(Image.open(path1)).unsqueeze(0).to(device)
        img2 = transform(Image.open(path2)).unsqueeze(0).to(device)
        
        match = siamese_model.predict_similarity(img1, img2, threshold=0.8485)
        distance = siamese_model.compute_distance(
            siamese_model.get_embedding(img1),
            siamese_model.get_embedding(img2)
        ).item()
        
        print(f"{path1} vs {path2}: Match={match}, Distance={distance:.4f}")
```

### Configuration Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input Size | 224×224 | Image height and width |
| Embedding Dimension | 256 | Size of the output embedding vector |
| Distance Metric | Euclidean (L2) | Metric used for comparing embeddings |
| Threshold | 0.8485 | Decision boundary for signature verification |
| Normalization Mean | [0.861, 0.861, 0.861] | Dataset-specific normalization |
| Normalization Std | [0.274, 0.274, 0.274] | Dataset-specific normalization |

## Installation & Setup

### Install Dependencies

```bash
pip install torch torchvision pillow huggingface_hub
```

### Quick Start

```python
# Download model from HuggingFace
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(
    repo_id="siddharth-magesh/siamese-signature-verification",
    filename="best_model.pth"
)

# Load and use
import torch
from modules.embedding_network import SimpleEmbeddingNetwork
from siamese_network import SiameseNetwork

embedding_net = SimpleEmbeddingNetwork(embedding_dim=256, input_size=(224, 224))
model = SiameseNetwork(embedding_net)
checkpoint = torch.load(weights_path, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
```

## Hyperparameter Tuning

This model was optimized using **Optuna** hyperparameter tuning framework. Key parameters and their optimized values:

| Parameter | Optimized Value | Range |
|-----------|-----------------|-------|
| Learning Rate | 0.000973 | 1e-5 to 1e-2 |
| Weight Decay | 0.000177 | 1e-6 to 1e-2 |
| Triplet Margin | 0.6836 | 0.1 to 2.0 |
| Scheduler Gamma | 0.3953 | 0.1 to 0.9 |
| Batch Size | 32 | 16 to 64 |

## Custom Architecture Implementation

The entire model was built from scratch without pre-trained weights:

### SimpleEmbeddingNetwork
- **Input:** Images of shape (B, 3, 224, 224)
- **Convolution Blocks:** 4 blocks with increasing channels (3 → 32 → 64 → 128 → 256)
- **Regularization:** BatchNorm, ReLU activations, MaxPooling, Dropout2d
- **Fully Connected Layers:** Feature flattening → 512 → 256 dimensions
- **Output:** L2-normalized 256-dimensional embeddings

### SiameseNetwork Wrapper
- **Shared Weights:** Same embedding network processes all input images
- **Triplet Loss:** $\mathcal{L} = \max(0, d(a,p) - d(a,n) + \text{margin})$
  - Where $a$ = anchor, $p$ = positive, $n$ = negative
  - $d(\cdot,\cdot)$ = Euclidean distance
- **Metric Learning:** Creates discriminative embedding space
- **Inference Methods:**
  - `get_embedding()`: Extract embedding for single or batch of images
  - `compute_distance()`: Calculate distance between embeddings
  - `predict_similarity()`: Threshold-based binary classification

## Evaluation Metrics Explained

- **Accuracy:** Percentage of correct classifications at threshold 0.8
- **Precision:** Of predicted genuine pairs, how many were actually genuine
- **Recall:** Of actual genuine pairs, how many were correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under the Receiver Operating Characteristic curve
- **Optimal Threshold:** Distance threshold that maximizes accuracy (0.8485)

## Confusion Matrix

```
                 Predicted Genuine  Predicted Forged
Actual Genuine         684 (TP)          416 (FN)
Actual Forged          316 (FP)          784 (TN)
```

## Limitations

- Model trained on specific signature dataset; may have domain bias
- Performance depends on image quality and signature consistency
- Works best with clear, full signatures (not partial or heavily degraded)
- Optimal threshold (0.8485) should be adjusted based on use case requirements

## Future Improvements

- Multi-user signature verification
- Real-time signature capture support
- Mobile deployment optimization
- Cross-domain signature adaptation

## Citation

If you use this model in your research, please cite:

```bibtex
@model{siamese_signature_verification_2025,
  title={Siamese Network for Signature Verification},
  author={Siddharth Magesh},
  year={2025},
  url={https://huggingface.co/siddharth-magesh/siamese-signature-verification}
}
```

## License

MIT License - See LICENSE file for details

## Contact & Support

For questions or issues, please open an issue on the model repository.

---

**Model Last Updated:** December 2025  
**Training Framework:** PyTorch  
**Dataset Source:** Kaggle (siddharthmagesh/signature-verfication)
