import torch
import torch.nn as nn


def print_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> None:
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    print(f"\nModel size: {total_size / 1e6:.2f} MB")
    print("=" * 60 + "\n")


def count_parameters_by_layer(model: nn.Module) -> dict:
    layer_params = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layer_type = type(module).__name__
            params = sum(p.numel() for p in module.parameters(recurse=False))
            
            if layer_type not in layer_params:
                layer_params[layer_type] = 0
            layer_params[layer_type] += params
    
    return layer_params
