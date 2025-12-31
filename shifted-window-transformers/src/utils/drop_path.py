import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # Work with diff tensor shapes; not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


if __name__ == "__main__":
    # Test drop path
    x = torch.randn(4, 196, 192)
    
    drop_path_layer = DropPath(drop_prob=0.1)
    drop_path_layer.train()
    
    y = drop_path_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Drop path module: {drop_path_layer}")
