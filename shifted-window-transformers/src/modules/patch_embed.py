import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = (
            img_size // patch_size,
            img_size // patch_size
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim) if not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape #x: (B, 3, H, W)
        assert (
            H == self.img_size and W == self.img_size
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x) # (B, 3, 224, 224) -> (B, 96, 56, 56)
        x = x.permute(0, 2, 3, 1).contiguous() # (B, 96, 56, 56) -> (B, 56, 56, 96)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    patch_embed = PatchEmbed(img_size=224, patch_size=4, in_channels=3, embed_dim=96)
    y = patch_embed(x)
    print(y.shape)