import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from typing import Optional


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    momentum: float = 0.9,
    betas: tuple = (0.9, 0.999),
    layer_decay: Optional[float] = None
) -> torch.optim.Optimizer:
    """
    Build optimizer for Swin Transformer.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adamw' or 'sgd')
        learning_rate: Base learning rate
        weight_decay: Weight decay factor
        momentum: Momentum for SGD
        betas: Beta coefficients for AdamW
        layer_decay: Layer-wise learning rate decay (optional)
        
    Returns:
        Configured optimizer
    """
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Don't apply weight decay to biases and LayerNorm
        if 'bias' in name or 'norm' in name or 'ln' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(param_groups, lr=learning_rate, betas=betas)
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(param_groups, lr=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


if __name__ == "__main__":
    # Test optimizer building
    from ..models import SwinTransformer
    from ..config import SwimConfig
    
    config = SwimConfig(image_size=224, num_classes=200)
    model = SwinTransformer(config)
    
    optimizer = build_optimizer(model, learning_rate=1e-4)
    print(f"Optimizer: {optimizer}")
    print(f"Number of param groups: {len(optimizer.param_groups)}")
