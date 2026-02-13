import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
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
	
class LightningAutoEncoder(L.LightningModule):
	def __init__(self, lr: float = 1e-3) -> None:
		super(LightningAutoEncoder).__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()
		self.lr = lr

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		z = self.encoder(x)
		x_hat = self.decoder(z)
		return x_hat
	
	def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
		x, _ = batch
		x = x.view(x.size(0), -1)
		x_hat = self(x)
		loss = F.mse_loss(x_hat, x)
		self.log("train_loss", loss)
		return loss
	
	def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
		x, _ = batch
		x = x.view(x.size(0), -1)
		x_hat = self(x)
		loss = F.mse_loss(x_hat, x)
		self.log("val_loss", loss)
		return loss
	
	def configure_optimizers(self) -> torch.optim.Optimizer:
		return torch.optim.Adam(params=self.parameters(), lr=self.lr)
	
transform = transforms.ToTensor()
train_dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
val_dataset = MNIST(os.getcwd(), train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

model = LightningAutoEncoder()
trainer = L.Trainer(
	max_epochs=5,
	accelerator="auto"
)
trainer.fit(
	model=model,
	train_loader=train_loader,
	val_loader=val_loader
)