import torch
import torch.nn as nn

def print_model_summary(model):
    print("\\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    
    total_params = 0
    trainable_params = 0
    
    print(f"{'Layer':<40} {'Parameters':<15} {'Trainable':<10}")
    print("-" * 70)
    
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        if parameter.requires_grad:
            trainable_params += params
        
        print(f"{name:<40} {params:>15,} {'Yes' if parameter.requires_grad else 'No':<10}")
    
    print("-" * 70)
    print(f"{'Total Parameters':<40} {total_params:>15,}")
    print(f"{'Trainable Parameters':<40} {trainable_params:>15,}")
    print(f"{'Non-trainable Parameters':<40} {(total_params - trainable_params):>15,}")
    print(f"{'Model Size (MB)':<40} {(total_params * 4 / 1024 / 1024):>15.2f}")
    print("=" * 70 + "\\n")


def count_parameters_by_layer(model):
    layer_params = {}
    
    for name, parameter in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        layer_params[layer_name] += parameter.numel()
    
    return layer_params
