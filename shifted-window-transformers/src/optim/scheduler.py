import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    LambdaLR,
    _LRScheduler
)
import math


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Learning rate starts from warmup_lr, increases linearly to base_lr during warmup,
    then follows a cosine decay to min_lr.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        warmup_lr: Initial learning rate during warmup
        min_lr: Minimum learning rate
        last_epoch: Last epoch index
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-5,
        last_epoch: int = -1
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + alpha * (base_lr - self.warmup_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_warmup",
    total_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-5,
    steps_per_epoch: int = None
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        steps_per_epoch: Steps per epoch (required for OneCycleLR)
        
    Returns:
        Configured scheduler
    """
    if scheduler_type == "cosine_warmup":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr
        )
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr
        )
    elif scheduler_type == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in optimizer.param_groups],
            total_steps=total_epochs * steps_per_epoch,
            pct_start=warmup_epochs / total_epochs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


if __name__ == "__main__":
    # Test scheduler
    import matplotlib.pyplot as plt
    
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=100)
    
    lrs = []
    for epoch in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    print(f"LR at epoch 0: {lrs[0]:.6f}")
    print(f"LR at epoch 10: {lrs[10]:.6f}")
    print(f"LR at epoch 50: {lrs[50]:.6f}")
    print(f"LR at epoch 99: {lrs[99]:.6f}")
