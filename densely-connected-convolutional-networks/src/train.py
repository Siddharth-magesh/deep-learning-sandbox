import os
import torch

from typing import Dict, Optional
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .optim.optimizer import build_optimizer_and_scheduler
from .utils.meters import AverageMeter

def train_one_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
	device: torch.device,
	scaler: Optional[GradScaler],
	epoch: int,
	cfg,
) -> Dict[str, float]:
	model.train()
	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	for step, (images, targets) in enumerate(dataloader):
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)
		with autocast(enabled=cfg.training.mixed_precision):
			outputs = model(images)
			loss = criterion(outputs, targets)
		if scaler is not None:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		if scheduler is not None and cfg.scheduler.step_per == "step":
			scheduler.step()
		with torch.no_grad():
			preds = outputs.argmax(dim=1)
			correct = (preds == targets).sum().item()
			acc = correct / targets.size(0)
		loss_meter.update(loss.item(), images.size(0))
		acc_meter.update(acc, images.size(0))

		if step % cfg.training.log_interval == 0:
			print(
				f"Epoch [{epoch}] "
				f"Step [{step}/{len(dataloader)}] "
				f"Loss: {loss_meter.avg:.4f} "
				f"Acc: {acc_meter.avg:.4f}"
			)
	return {
		"loss": loss_meter.avg,
		"accuracy": acc_meter.avg
	}

@torch.no_grad()
def validate(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
) -> Dict[str, float]:
	model.eval()
	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	for images, targets in dataloader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		outputs = model(images)
		loss = criterion(outputs, targets)
		preds = outputs.argmax(dim=1)
		correct = (preds == targets).sum().item()
		acc = correct / targets.size(0)
		loss_meter.update(loss.item(), images.size(0))
		acc_meter.update(acc, images.size(0))

	return {
		"val_loss": loss_meter.avg,
		"val_accuracy": acc_meter.avg
	}

def train(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: Optional[DataLoader],
	cfg,
) -> None:
	device = torch.device(cfg.training.device)
	model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer, scheduler = build_optimizer_and_scheduler(
		model,
		cfg.optimizer,
		cfg.scheduler,
	)
	scaler = GradScaler(enabled=cfg.training.mixed_precision)
	best_val_acc = 0.0
	for epoch in range(1, cfg.training.epochs + 1):
		train_metrics = train_one_epoch(
			model=model,
			dataloader=train_loader,
			criterion=criterion,
			optimizer=optimizer,
			scheduler=scheduler,
			device=device,
			scaler=scaler,
			epoch=epoch,
			cfg=cfg,
		)
		if scheduler is not None and cfg.scheduler.step_per == "epoch":
			scheduler.step()
		print(
            		f"[Epoch {epoch}] "
            		f"Train Loss: {train_metrics['loss']:.4f}, "
            		f"Train Acc: {train_metrics['accuracy']:.4f}"
        	)
		if val_loader is not None:
			val_metrics = validate(
				model=model,
				dataloader=val_loader,
				criterion=criterion,
				device=device,
			)

			print(
				f"[Epoch {epoch}] "
				f"Val Loss: {val_metrics['val_loss']:.4f}, "
				f"Val Acc: {val_metrics['val_accuracy']:.4f}"
			)
			if val_metrics["val_accuracy"] > best_val_acc:
				best_val_acc = val_metrics["val_accuracy"]
				save_checkpoint(
					model,
					optimizer,
					epoch,
					best_val_acc,
					cfg,
					filename="best.pth",
				)

def save_checkpoint(
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	best_acc: float,
	cfg,
	filename: str,
) -> None:
	os.makedirs(cfg.runtime.output_dir, exist_ok=True)

	path = os.path.join(cfg.runtime.output_dir, filename)

	torch.save(
		{
			"epoch": epoch,
			"model_state": model.state_dict(),
			"optimizer_state": optimizer.state_dict(),
			"best_acc": best_acc,
			"config": cfg,
		},
		path,
	)

