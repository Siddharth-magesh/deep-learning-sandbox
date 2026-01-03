import torch
import torch.nn as nn
from typing import Dict
from torch.utils.data import DataLoader

from .utils.meters import AverageMeter


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    cfg,
) -> Dict[str, float]:

    device = torch.device(cfg.training.device)
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top5_acc_meter = AverageMeter()
    
    print("Evaluating model...")
    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        acc = correct / targets.size(0)

        if cfg.model.num_classes > 5:
            _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            top5_correct = top5_preds.eq(targets.view(-1, 1).expand_as(top5_preds)).sum().item()
            top5_acc = top5_correct / targets.size(0)
            top5_acc_meter.update(top5_acc, images.size(0))
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        
        if step % 10 == 0:
            print(f"Step [{step}/{len(dataloader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"Acc: {acc_meter.avg:.4f}")

    results = {
        "test_loss": loss_meter.avg,
        "test_accuracy": acc_meter.avg,
    }
    
    if cfg.model.num_classes > 5:
        results["test_top5_accuracy"] = top5_acc_meter.avg
    
    return results


@torch.no_grad()
def evaluate_single_image(
    model: nn.Module,
    image: torch.Tensor,
    cfg,
) -> Dict[str, any]:

    device = torch.device(cfg.training.device)
    model.to(device)
    model.eval()
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    
    top5_probs, top5_indices = probs.topk(5, dim=1, largest=True, sorted=True)
    
    return {
        "predicted_class": top5_indices[0, 0].item(),
        "confidence": top5_probs[0, 0].item(),
        "top5_classes": top5_indices[0].cpu().tolist(),
        "top5_probs": top5_probs[0].cpu().tolist(),
    }
