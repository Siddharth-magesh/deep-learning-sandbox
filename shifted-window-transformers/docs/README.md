# Swin Transformer Documentation

Complete documentation for the Swin Transformer (Shifted Window Transformer) implementation.

## Contents

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Detailed architecture explanation |
| [QUICK_START.md](QUICK_START.md) | Getting started guide |
| [TRAINING.md](TRAINING.md) | Training configuration and tips |
| [API_REFERENCE.md](API_REFERENCE.md) | Module and class reference |
| [DATA.md](DATA.md) | Dataset and data loading |

## Overview

Swin Transformer is a hierarchical vision transformer that uses shifted windows for efficient self-attention computation. This implementation follows the original paper:

> **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**  
> Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo  
> [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)

## Key Features

- Hierarchical feature maps (like CNNs)
- Linear computational complexity with respect to image size
- Shifted window mechanism for cross-window connections
- Compatible with dense prediction tasks (detection, segmentation)

## Model Variants

| Model | Embed Dim | Depths | Heads | Parameters |
|-------|-----------|--------|-------|------------|
| Swin-Tiny | 96 | [2,2,6,2] | [3,6,12,24] | 28M |
| Swin-Small | 96 | [2,2,18,2] | [3,6,12,24] | 50M |
| Swin-Base | 128 | [2,2,18,2] | [4,8,16,32] | 88M |

## Quick Links

- [Training Guide](TRAINING.md)
- [Architecture Details](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
