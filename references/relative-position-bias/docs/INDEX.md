# Documentation Index

Welcome to the Relative Position Bias documentation. This guide will help you understand, implement, and use relative position bias in attention-based models.

## Table of Contents

### Getting Started

1. **[README](../README.md)**
   - Project overview
   - Quick start guide
   - Installation instructions
   - Basic usage

### Core Documentation

2. **[Concepts](CONCEPTS.md)**
   - What is relative position bias?
   - Why use it over absolute positions?
   - 1D vs 2D relative position bias
   - Mathematical foundations
   - Key properties and benefits

3. **[Architecture](ARCHITECTURE.md)**
   - System architecture overview
   - Module hierarchy
   - Component descriptions
   - Data flow
   - Design decisions
   - Parameter counts
   - Computational complexity

4. **[Implementation](IMPLEMENTATION.md)**
   - Detailed implementation guide
   - Indexing algorithms
   - Bias table lookup
   - Attention integration
   - Memory efficiency
   - Numerical stability
   - Shape transformations
   - Optimization techniques

### Practical Guides

5. **[API Reference](API_REFERENCE.md)**
   - Complete API documentation
   - Function signatures
   - Parameter descriptions
   - Return values
   - Code examples for each component

6. **[Training Guide](TRAINING_GUIDE.md)**
   - Setup and configuration
   - Training from scratch
   - Hyperparameter tuning
   - Monitoring and debugging
   - Common issues and solutions
   - Advanced techniques
   - Performance optimization

7. **[Examples](EXAMPLES.md)**
   - Basic usage examples
   - Training examples
   - Visualization examples
   - Configuration examples
   - Advanced use cases
   - Inference examples
   - Debugging examples

## Quick Navigation

### By Topic

**Understanding Relative Position Bias**
- Start with [Concepts](CONCEPTS.md) for theoretical background
- Review [Architecture](ARCHITECTURE.md) for system design
- Check [Implementation](IMPLEMENTATION.md) for technical details

**Using the Code**
- Begin with [README](../README.md) for quick start
- Refer to [API Reference](API_REFERENCE.md) for function details
- Follow [Examples](EXAMPLES.md) for common use cases

**Training Models**
- Read [Training Guide](TRAINING_GUIDE.md) for complete workflow
- Use [Examples](EXAMPLES.md) for training code
- Reference [API Reference](API_REFERENCE.md) for specific functions

### By Skill Level

**Beginners**
1. [README](../README.md) - Installation and quick start
2. [Concepts](CONCEPTS.md) - Understand the basics
3. [Examples](EXAMPLES.md) - See working code

**Intermediate**
1. [Architecture](ARCHITECTURE.md) - System design
2. [Training Guide](TRAINING_GUIDE.md) - Train your models
3. [API Reference](API_REFERENCE.md) - Detailed API

**Advanced**
1. [Implementation](IMPLEMENTATION.md) - Deep dive into code
2. [Training Guide](TRAINING_GUIDE.md) - Advanced techniques
3. [Examples](EXAMPLES.md) - Complex use cases

## Key Concepts Overview

### Relative Position Bias

A mechanism to encode positional information based on relative distances between positions rather than absolute locations.

**Benefits**:
- Length generalization
- Translation invariance
- Parameter efficiency
- Better inductive bias

### 1D vs 2D Bias

**1D Bias**: For sequential data (text, time series)
- Relative position: i - j
- Parameter count: 2N - 1 per head

**2D Bias**: For spatial data (images, grids)
- Relative position: (i_h - j_h, i_w - j_w)
- Parameter count: (2W_h - 1) × (2W_w - 1) per head

### Core Components

1. **RelativePositionBias**: Generates position-dependent bias
2. **MultiHeadAttention**: Attention with RPB support
3. **TransformerBlock**: Complete transformer layer
4. **VisionTransformer**: Full model implementation

## Code Organization

```
relative-position-bias/
├── configs/              # Configuration files
├── data/                 # Dataset and data loading
├── models/               # Model architectures
├── modules/              # Core components (attention, bias)
├── experiments/          # Training, evaluation, visualization
├── docs/                 # Documentation (you are here)
├── main.py              # Training entry point
├── demo.py              # Demonstration script
└── README.md            # Project overview
```

## Common Tasks

### How do I...

**...understand what relative position bias is?**
→ Read [Concepts](CONCEPTS.md)

**...implement a model with RPB?**
→ Check [Examples](EXAMPLES.md) - "Building a Vision Transformer"

**...train a model?**
→ Follow [Training Guide](TRAINING_GUIDE.md)

**...visualize attention patterns?**
→ See [Examples](EXAMPLES.md) - "Visualization Examples"

**...debug issues?**
→ Refer to [Training Guide](TRAINING_GUIDE.md) - "Common Issues"

**...understand the code internals?**
→ Read [Implementation](IMPLEMENTATION.md)

**...find a specific function?**
→ Use [API Reference](API_REFERENCE.md)

**...optimize performance?**
→ See [Training Guide](TRAINING_GUIDE.md) - "Performance Optimization"

## Additional Resources

### Papers

- **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- **T5**: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- **ALiBi**: Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"

### Related Projects

- Hugging Face Transformers
- PyTorch Vision
- timm (PyTorch Image Models)

## Contributing

For bug reports, feature requests, or contributions, please see the project repository.

## License

This project is licensed under the MIT License.

## Support

If you encounter any issues or have questions:
1. Check the [Training Guide](TRAINING_GUIDE.md) for common issues
2. Review [Examples](EXAMPLES.md) for similar use cases
3. Consult [API Reference](API_REFERENCE.md) for function details
4. Read [Implementation](IMPLEMENTATION.md) for technical details

## Version History

- **v1.0.0** - Initial release
  - 1D and 2D relative position bias
  - Multi-head attention with RPB
  - Vision Transformer implementation
  - Training and visualization utilities
  - Complete documentation
