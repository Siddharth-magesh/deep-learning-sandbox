import torch
import torch.nn as nn
from .dense_layer import DenseLayer

class DenseBlock(nn.Module):
	def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int, dropout: float = 0.0) -> None:
		super(DenseBlock, self).__init__()
		self.layers = nn.ModuleList()
		channels = in_channels
		for i in range(num_layers):
			self.layers.append(
				DenseLayer(
					in_channels=channels,
					growth_rate=growth_rate,
					bn_size=bn_size,
					dropout=dropout
				)
			)
			channels += growth_rate
		self.out_channels = channels

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for layer in self.layers:
			x = layer(x)
		return x
