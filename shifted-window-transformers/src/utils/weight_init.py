import torch
import torch.nn as nn
import math


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """
    Truncated normal initialization.
    
    Fills the input Tensor with values drawn from a truncated normal distribution.
    The values are effectively drawn from the normal distribution N(mean, std^2) 
    with values outside [a, b] redrawn until they are within the bounds.
    
    Args:
        tensor: Input tensor to fill
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        a: Minimum cutoff value
        b: Maximum cutoff value
        
    Returns:
        Tensor filled with truncated normal values
    """
    with torch.no_grad():
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        
        if (mean < a - 2 * std) or (mean > b + 2 * std):
            print(f"Warning: mean {mean} is more than 2 std from [{a}, {b}] in trunc_normal_. "
                  "The distribution of values may be incorrect.")
        
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        
        return tensor


def init_weights_vit(module: nn.Module, name: str = '', head_bias: float = 0.0) -> None:
    """
    Weight initialization for Vision Transformer / Swin Transformer.
    
    Args:
        module: Module to initialize
        name: Name of the module (for special handling)
        head_bias: Bias initialization for the classification head
    """
    if isinstance(module, nn.Linear):
        if 'head' in name:
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        # Patch embedding uses conv2d
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_swin(model: nn.Module) -> None:
    """
    Initialize weights for Swin Transformer model.
    
    Args:
        model: Swin Transformer model
    """
    for name, module in model.named_modules():
        init_weights_vit(module, name)


if __name__ == "__main__":
    # Test weight initialization
    linear = nn.Linear(192, 768)
    print(f"Before init - weight mean: {linear.weight.mean():.4f}, std: {linear.weight.std():.4f}")
    
    trunc_normal_(linear.weight, std=0.02)
    print(f"After init - weight mean: {linear.weight.mean():.4f}, std: {linear.weight.std():.4f}")
