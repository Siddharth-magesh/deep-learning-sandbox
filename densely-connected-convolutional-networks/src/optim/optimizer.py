import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..config.optim_config import OptimizerConfig, SchedulerConfig

def build_optimizer(
	model: nn.Module,
	cfg: OptimizerConfig,
) -> torch.optim.Optimizer:
	params = model.parameters()
	name = cfg.name.lower()

	if name == "sgd":
		optimizer = torch.optim.SGD(
			params,
			lr=cfg.lr,
			momentum=cfg.momentum,
			weight_decay=cfg.weight_decay
		)
	elif name == "adam":
		optimizer = torch.optim.Adam(
			params,
			lr=cfg.lr,
			betas=cfg.betas,
			weight_decay=cfg.weight_decay
		)
	elif name == "adamw":
		optimizer = torch.optim.AdamW(
			params,
			lr=cfg.lr,
			betas=cfg.betas,
			weight_decay=cfg.weight_decay
		)
	else:
		raise ValueError(f"Unsupported optimizer: {cfg.name}")

	return optimizer

def build_scheduler(
	optimizer: torch.optim.Optimizer,
	cfg: SchedulerConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
	if cfg.name is None or cfg.name.lower() == "none":
		return None

	name = cfg.name.lower()
	if name == "step":
		scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=cfg.step_size,
			gamma=cfg.gamma
		)

	elif name == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=cfg.t_max,
		)
	else:
		raise ValueError(f"Unsupported scheduler: {cfg.name}")
	return scheduler

def build_optimizer_and_scheduler(
	model: nn.Module,
	optim_cfg: OptimizerConfig,
	sched_cfg: SchedulerConfig
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
	optimizer = build_optimizer(model, optim_cfg)
	scheduler = build_scheduler(optimizer, sched_cfg)

	return optimizer, scheduler
