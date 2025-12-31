```
shifted-window-transformers/
├── src/
│   ├── config/
│   │   └── swin_tiny.py              # Model configuration (constants only)
│   │
│   ├── data/
│   │   └── tiny_imagenet.py          # Dataset + transforms
│   │
│   ├── modules/                      # 🔧 Reusable building blocks
│   │   ├── __init__.py
│   │   ├── patch_embed.py            # Patch embedding (Conv2d-based)
│   │   ├── window_ops.py             # window_partition & window_reverse
│   │   ├── attention.py              # WindowAttention (W-MSA & SW-MSA)
│   │   ├── mlp.py                    # Feed-forward network
│   │   ├── swin_block.py             # SwinTransformerBlock
│   │   ├── patch_merge.py            # PatchMerging
│   │
│   ├── models/                       # 🧠 Full models (architecture assembly)
│   │   ├── __init__.py
│   │   ├── swin_stage.py             # One hierarchical stage
│   │   └── swin_tiny.py              # Full Swin-Tiny model
│   │
│   ├── utils/
│   │   ├── drop_path.py              # Stochastic depth
│   │   └── weight_init.py            # Init helpers
│   │
│   ├── train.py                      # Training loop
│   ├── eval.py                       # Evaluation
│   └── test_shapes.py                # Shape sanity tests
│
├── requirements.txt
└── README.md
```