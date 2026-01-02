from dataclasses import dataclass, field
from typing import List

@dataclass
class DenseNetConfig:
	name: str = "densenet201"
	num_classes: int = 1000
	growth_rate: int = 32
	block_layers: List[int] = field(
		default_factory=lambda: [6, 12, 48, 32]
	)
	bn_size: int = 4
	compression_factor: float = 0.5
	dropout: float = 0.0
	in_channels: int = 3
	input_size: int = 224
