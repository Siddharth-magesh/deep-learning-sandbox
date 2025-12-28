import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR, MultiStepLR
import math

def build_scheduler(optimizer, config, steps_per_epoch):
    scheduler_type = getattr(config, 'scheduler_type', 'cosine_warmup').lower()
    
    if scheduler_type == 'cosine_warmup':
        total_steps = steps_per_epoch * config.max_epochs
        warmup_steps = steps_per_epoch * config.warmup_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=getattr(config, 'min_lr', 1e-6)
        )
    
    elif scheduler_type == 'step':
        step_size = getattr(config, 'step_size', 30)
        gamma = getattr(config, 'gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        milestones = getattr(config, 'milestones', [30, 60, 90])
        gamma = getattr(config, 'gamma', 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'constant':
        scheduler = LambdaLR(optimizer, lambda step: 1.0)
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    return scheduler
