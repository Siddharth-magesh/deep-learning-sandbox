# Documentation Index

Welcome to the DenseNet (Densely Connected Convolutional Networks) documentation!

## 📚 Documentation Structure

This documentation is organized to help you get started quickly and dive deep into specific topics as needed.

### For Beginners

1. **[README.md](../README.md)** - Start here!
   - Project overview
   - Features and highlights
   - Installation instructions
   - Quick start examples

2. **[QUICK_START.md](QUICK_START.md)** - Get up and running fast
   - Installation guide
   - Basic usage examples
   - Command line interface
   - Configuration basics
   - Tips for best results

### For Understanding

3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Deep dive into the architecture
   - Dense connectivity explained
   - Mathematical foundations
   - Network structure
   - Implementation details
   - Performance characteristics

### For Training

4. **[TRAINING.md](TRAINING.md)** - Advanced training techniques
   - Complete training pipeline
   - Hyperparameter tuning
   - Optimization strategies
   - Troubleshooting guide
   - Best practices

### For Developers

5. **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
   - All classes and functions
   - Parameters and return types
   - Usage examples
   - Code snippets

## 🎯 Quick Navigation

### I want to...

**...get started quickly**
→ Go to [README.md](../README.md) and [QUICK_START.md](QUICK_START.md)

**...understand how DenseNet works**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md)

**...train a model**
→ Check [QUICK_START.md](QUICK_START.md) for basics, [TRAINING.md](TRAINING.md) for advanced

**...customize the implementation**
→ Use [API_REFERENCE.md](API_REFERENCE.md)

**...troubleshoot an issue**
→ See the troubleshooting section in [TRAINING.md](TRAINING.md)

**...understand the code**
→ Read [API_REFERENCE.md](API_REFERENCE.md)

## 📖 Recommended Reading Order

### For First-Time Users

1. [README.md](../README.md) - Overview and installation
2. [QUICK_START.md](QUICK_START.md) - Run your first training
3. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand what you're running
4. [TRAINING.md](TRAINING.md) - Optimize your results

### For Researchers

1. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the theory
2. [API_REFERENCE.md](API_REFERENCE.md) - See the implementation
3. [TRAINING.md](TRAINING.md) - Experimental setup
4. [QUICK_START.md](QUICK_START.md) - Quick reference

### For Engineers

1. [QUICK_START.md](QUICK_START.md) - Get it running
2. [API_REFERENCE.md](API_REFERENCE.md) - Integration details
3. [TRAINING.md](TRAINING.md) - Optimization and deployment
4. [ARCHITECTURE.md](ARCHITECTURE.md) - Deep understanding

## 🔑 Key Concepts

### DenseNet Architecture

DenseNet introduces **dense connectivity** where each layer is connected to every other layer in a feed-forward fashion. This creates L(L+1)/2 connections in a network with L layers.

**Benefits:**
- Alleviates vanishing gradient problem
- Encourages feature reuse
- Reduces number of parameters
- Improves accuracy

### Main Components

1. **Dense Block** - Stack of densely connected layers
2. **Dense Layer** - Bottleneck layer with 1×1 and 3×3 convolutions
3. **Transition Layer** - Reduces dimensions between blocks
4. **Growth Rate** - Number of feature maps added per layer

### Model Variants

- **DenseNet-121**: 6.96M parameters, good for CIFAR
- **DenseNet-169**: 14M parameters, balanced performance
- **DenseNet-201**: 20M parameters, high accuracy
- **DenseNet-264**: 34M parameters, maximum capacity

## 💡 Common Tasks

### Training on CIFAR-10

```bash
uv run python src/main.py --mode train --model densenet121 --dataset cifar10
```

See: [QUICK_START.md](QUICK_START.md#1-training-on-cifar-10)

### Training on ImageNet

Requires downloaded ImageNet dataset:

```bash
uv run python src/main.py --mode train --model densenet201 --dataset imagenet
```

See: [TRAINING.md](TRAINING.md#imagenet)

### Custom Configuration

```python
from src.config.config import Config

cfg = Config()
cfg.model.growth_rate = 24  # Reduce for smaller model
cfg.training.epochs = 100
```

See: [API_REFERENCE.md](API_REFERENCE.md#configuration)

### Evaluation

```bash
uv run python src/main.py --mode eval --checkpoint outputs/best.pth
```

See: [QUICK_START.md](QUICK_START.md#3-evaluation)

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution | Reference |
|-------|----------|-----------|
| Out of Memory | Reduce batch size | [TRAINING.md](TRAINING.md#issue-1-loss-is-nan) |
| Slow training | Enable mixed precision | [TRAINING.md](TRAINING.md#mixed-precision-training) |
| Poor accuracy | Adjust hyperparameters | [TRAINING.md](TRAINING.md#hyperparameter-tuning) |
| Import errors | Check installation | [README.md](../README.md#installation) |

## 📊 Performance Benchmarks

### CIFAR-10

| Model | Parameters | Accuracy | Training Time |
|-------|-----------|----------|---------------|
| DenseNet-121 | 6.96M | ~95% | 5-10 hours |
| DenseNet-169 | 14M | ~95.5% | 8-15 hours |
| DenseNet-201 | 20M | ~96% | 12-20 hours |

### ImageNet

| Model | Parameters | Top-1 | Top-5 | Training Time |
|-------|-----------|-------|-------|---------------|
| DenseNet-121 | ~8M | ~74% | ~92% | 3-4 days |
| DenseNet-169 | ~14M | ~76% | ~93% | 4-5 days |
| DenseNet-201 | ~20M | ~77% | ~93.5% | 5-7 days |

See: [ARCHITECTURE.md](ARCHITECTURE.md#performance)

## 🔬 Research & Citations

### Original Paper

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In CVPR.

**Paper**: https://arxiv.org/abs/1608.06993

### BibTeX

```bibtex
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}
```

## 🛠️ Implementation Details

### Technology Stack

- **Framework**: PyTorch 2.0+
- **Language**: Python 3.8+
- **Package Manager**: UV (recommended) or pip
- **Dependencies**: torch, torchvision, numpy, pillow

### Project Structure

```
densely-connected-convolutional-networks/
├── src/              # Source code
├── docs/             # Documentation (you are here!)
├── test_components.py # Tests
└── pyproject.toml    # Project config
```

See: [README.md](../README.md#project-structure)

## 📝 Contributing

We welcome contributions! Please:

1. Read the documentation to understand the codebase
2. Check [API_REFERENCE.md](API_REFERENCE.md) for implementation details
3. Follow the existing code style
4. Add tests for new features
5. Update documentation as needed

## 🆘 Getting Help

If you're stuck:

1. Check the relevant documentation page
2. Look for similar issues in [TRAINING.md](TRAINING.md#common-issues-and-solutions)
3. Review the [API_REFERENCE.md](API_REFERENCE.md)
4. Run the test suite: `uv run python test_components.py`

## 📚 External Resources

### Learning Resources

- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **DenseNet Paper**: https://arxiv.org/abs/1608.06993
- **Original Implementation**: https://github.com/liuzhuang13/DenseNet

### Datasets

- **CIFAR-10/100**: https://www.cs.toronto.edu/~kriz/cifar.html
- **ImageNet**: https://www.image-net.org/

## 🔄 Updates

This documentation is actively maintained. Last updated: 2026-01-03

---

**Ready to get started?** → Go to [QUICK_START.md](QUICK_START.md)

**Want to understand the architecture?** → Read [ARCHITECTURE.md](ARCHITECTURE.md)

**Need API details?** → Check [API_REFERENCE.md](API_REFERENCE.md)
