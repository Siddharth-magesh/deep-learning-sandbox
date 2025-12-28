import torch
from torch.optim import AdamW, SGD, Adam

def build_optimizer(model, config):
    optimizer_type = getattr(config, 'optimizer_type', 'adamw').lower()
    
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon
        )
    elif optimizer_type == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon
        )
    elif optimizer_type == 'sgd':
        momentum = getattr(config, 'momentum', 0.9)
        nesterov = getattr(config, 'nesterov', True)
        optimizer = SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=momentum,
            nesterov=nesterov
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    return optimizer
