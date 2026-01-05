import torch

class IndexPE:
	def __init__(self, seq_len: int):
		self.seq_len = seq_len

	def forward(self):
		return torch.arange(self.seq_len).unsqueeze(1)
