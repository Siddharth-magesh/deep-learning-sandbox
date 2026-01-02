import torch
import torch.nn as nn

class TransitionLayer(nn.Module):
	def __init__(self, in_channels: int, compression_factor: float = 0.5) -> None:
		super(TransitionLayer, self).__init__()
		assert 0 < compression_factor <= 1.0 , "compression_factor must be in (0,1]"
		out_channels = 	int(in_channels * compression_factor)
		self.layers = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(
				in_channels=in_channels,
				out_channels=out_channels,
				stride=1,
				kernel_size=1,
				bias=False
			),
			nn.AvgPool2d(kernel_size=2, stride=2)
		)
		self.out_channels = out_channels

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)
