import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config.model_config import DenseNetConfig
from ..modules.dense_block import DenseBlock
from ..modules.transition_layer import TransitionLayer

class DenseNet(nn.Module):
	def __init__(self, cfg: DenseNetConfig) -> None:
		super(DenseNet, self).__init__()
		self.cfg = cfg
		growth_rate = cfg.growth_rate
		block_layers = cfg.block_layers
		bn_size = cfg.bn_size
		compression_factor = cfg.compression_factor
		dropout = cfg.dropout

		self.stem = nn.Sequential(
			nn.Conv2d(
				in_channels=cfg.in_channels,
				out_channels=64,
				kernel_size=7,
				stride=2,
				padding=3,
				bias=False
			),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		)
		channels = 64
		self.blocks = nn.ModuleList()

		for i, num_layers in enumerate(block_layers):
			block = DenseBlock(
				num_layers=num_layers,
				in_channels=channels,
				growth_rate=growth_rate,
				bn_size=bn_size,
				dropout=dropout
			)
			self.blocks.append(block)
			channels = block.out_channels

			if i != len(block_layers) - 1:
				transition = TransitionLayer(
					in_channels=channels,
					compression_factor=compression_factor
				)
				self.blocks.append(transition)
				channels = transition.out_channels

		self.final_norm = nn.BatchNorm2d(channels)
		self.classifier = nn.Linear(
			in_features = channels,
			out_features = cfg.num_classes
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem(x)
		for block in self.blocks:
			x = block(x)
		x = self.final_norm(x)
		x = F.relu(x, inplace=True)
		x = F.adaptive_avg_pool2d(x, (1,1))
		x = torch.flatten(x, 1)
		return self.classifier(x)
