import torch
import torch.nn as nn
from typing import List, Optional

from ..modules import PatchEmbed, SwinTransformerBlock, PatchMerging
from ..config import SwimConfig


class SwinStage(nn.Module):
    """A single Swin Transformer stage consisting of multiple Swin blocks and optional downsampling."""
    
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | List[float] = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: bool = True
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Handle drop_path rates
        if isinstance(drop_path, (list, tuple)):
            drop_path_rates = drop_path
        else:
            drop_path_rates = [drop_path] * depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            block = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer
            )
            self.blocks.append(block)

        if downsample:
            self.downsample = PatchMerging(
                input_resolution=input_resolution,
                dim=dim,
                norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
    
    Paper: https://arxiv.org/abs/2103.14030
    
    Args:
        config: SwimConfig containing all model hyperparameters
    """
    
    def __init__(self, config: SwimConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.num_layers = len(config.depths)
        self.embed_dim = config.embed_dim
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(config.embed_dim * 2 ** i_layer)
            layer_resolution = (
                patches_resolution[0] // (2 ** i_layer),
                patches_resolution[1] // (2 ** i_layer)
            )
            
            stage = SwinStage(
                dim=layer_dim,
                input_resolution=layer_resolution,
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=(i_layer < self.num_layers - 1)
            )
            self.stages.append(stage)
        
        # Final norm and classifier
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, config.num_classes) if config.num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        x = self.patch_embed(x)  # (B, H, W, C)
        x = self.pos_drop(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.norm(x)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = x.flatten(2)  # (B, C, H*W)
        x = self.avgpool(x)  # (B, C, 1)
        x = x.flatten(1)  # (B, C)
        return x
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, C, H, W)
            labels: Optional labels for computing loss
            
        Returns:
            logits: Classification logits
            loss: Cross-entropy loss if labels provided, else None
        """
        x = self.forward_features(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
        
        return logits, loss


def swin_tiny(num_classes: int = 200, image_size: int = 224) -> SwinTransformer:
    """Swin-Tiny model configuration."""
    config = SwimConfig(
        image_size=image_size,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    return SwinTransformer(config)


def swin_small(num_classes: int = 200, image_size: int = 224) -> SwinTransformer:
    """Swin-Small model configuration."""
    config = SwimConfig(
        image_size=image_size,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7
    )
    return SwinTransformer(config)


def swin_base(num_classes: int = 200, image_size: int = 224) -> SwinTransformer:
    """Swin-Base model configuration."""
    config = SwimConfig(
        image_size=image_size,
        num_classes=num_classes,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7
    )
    return SwinTransformer(config)


if __name__ == "__main__":
    # Test the model
    config = SwimConfig(
        image_size=224,
        num_classes=200,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24]
    )
    model = SwinTransformer(config)
    
    x = torch.randn(2, 3, 224, 224)
    logits, _ = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
