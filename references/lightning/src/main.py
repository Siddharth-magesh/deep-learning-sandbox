import os
import torch
import torch.nn as nn
import torch.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data imprt DataLoader
import lightning as L

class Encoder(nn.Module):
	def __init__(self) -> None:
		super(Encoder).__init__()
		self.l1 = nn.Sequential(
			nn.Linear(
				in_features=28*28,
				out_features=64
			),
			nn.ReLU(),
			nn.Linear(
				in_features=64,
				out_features=3
			)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.l1(x)

class Decoder(nn.Module):
	def __init__(self) -> None:
		super(Decoder).__init__()
		self.l1 = nn.Sequential(
			nn.Linear(
				in_features=3,
				out_features=64
			),
			nn.ReLU(),
			nn.Linear(
				in_features=64,
				out_features=28*28
			)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.l1(x)
