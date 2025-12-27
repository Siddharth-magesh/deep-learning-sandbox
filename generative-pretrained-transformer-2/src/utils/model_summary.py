import torch
import torch.nn as nn


def print_model_summary(model: nn.Module, input_size: tuple = (1, 128)):
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Non-trainable parameters: {(total_params - trainable_params):,}")
    
    print("\n" + "-" * 80)
    print(f"{'Layer':<40} {'Output Shape':<25} {'Params':<15}")
    print("-" * 80)
    
    device = next(model.parameters()).device
    dummy_input = torch.randint(0, model.config.vocab_size, input_size, device=device)
    
    hooks = []
    layer_info = []
    
    def hook_fn(module, input, output):
        class_name = module.__class__.__name__
        if isinstance(output, tuple):
            output_shape = str(output[0].shape) if len(output) > 0 else "N/A"
        else:
            output_shape = str(output.shape) if hasattr(output, 'shape') else "N/A"
        
        num_params = sum(p.numel() for p in module.parameters())
        layer_info.append((class_name, output_shape, num_params))
    
    for name, layer in model.named_modules():
        if len(list(layer.children())) == 0:
            hooks.append(layer.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    seen = set()
    for layer_name, output_shape, num_params in layer_info:
        if (layer_name, num_params) not in seen:
            seen.add((layer_name, num_params))
            params_str = f"{num_params:,}" if num_params > 0 else "-"
            print(f"{layer_name:<40} {output_shape:<25} {params_str:<15}")
    
    print("-" * 80)
    
    param_size = total_params * 4 / (1024 ** 2)
    print(f"\nEstimated model size: {param_size:.2f} MB (FP32)")
    print(f"Estimated model size: {param_size / 2:.2f} MB (FP16)")
    
    print("\nLayer breakdown:")
    embedding_params = sum(p.numel() for n, p in model.named_parameters() if 'embedding' in n)
    attention_params = sum(p.numel() for n, p in model.named_parameters() if 'attention' in n or 'query' in n or 'key' in n or 'value' in n)
    feedforward_params = sum(p.numel() for n, p in model.named_parameters() if 'feedforward' in n or 'linear' in n)
    layernorm_params = sum(p.numel() for n, p in model.named_parameters() if 'layer_norm' in n or 'norm' in n)
    
    print(f"  Embeddings: {embedding_params:,} ({embedding_params / 1e6:.2f}M) - {embedding_params / total_params * 100:.1f}%")
    print(f"  Attention: {attention_params:,} ({attention_params / 1e6:.2f}M) - {attention_params / total_params * 100:.1f}%")
    print(f"  Feed-Forward: {feedforward_params:,} ({feedforward_params / 1e6:.2f}M) - {feedforward_params / total_params * 100:.1f}%")
    print(f"  Layer Norm: {layernorm_params:,} ({layernorm_params / 1e6:.2f}M) - {layernorm_params / total_params * 100:.1f}%")
    
    print("\n" + "=" * 80 + "\n")


def count_parameters_by_layer(model: nn.Module):
    print("\nDetailed parameter count by layer:")
    print("-" * 60)
    print(f"{'Layer Name':<45} {'Parameters':>15}")
    print("-" * 60)
    
    for name, param in model.named_parameters():
        print(f"{name:<45} {param.numel():>15,}")
    
    print("-" * 60)
    print(f"{'Total':<45} {sum(p.numel() for p in model.parameters()):>15,}")
    print("-" * 60 + "\n")
