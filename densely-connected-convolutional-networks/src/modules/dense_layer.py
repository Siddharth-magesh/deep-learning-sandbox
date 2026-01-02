import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
	def __init__(
		self,
		in_channels: int,
		growth_rate: int,
		bn_size: int,
		dropout: float = 0.0,
	) -> None:
		super(DenseLayer, self).__init__()
		#Bottleneck
		self.dimension_reduction = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(
				in_channels=in_channels,
				out_channels=bn_size * growth_rate,
				kernel_size=1,
				stride=1,
				bias=False
			)
		)
		#Feature Extraction
		self.feature_extraction = nn.Sequential(
			nn.BatchNorm2d(bn_size * growth_rate),
			nn.ReLU(inplace=True),
			nn.Conv2d(
				in_channels=bn_size * growth_rate,
				out_channels=growth_rate,
				kernel_size=3,
				stride=1,
				padding=1,
				bias=False
			)
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.dimension_reduction(x)
		out = self.feature_extraction(out)
		out = torch.cat((x, out), dim=1)
		return out
