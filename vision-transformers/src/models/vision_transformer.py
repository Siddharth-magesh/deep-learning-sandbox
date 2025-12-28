import torch
import torch.nn as nn
from ..modules import PatchEmbedding, TransformerEncoder, MultiHeadPerceptronClassifier
from ..config import ViTConfig

class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        
        self.patch_embedding = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embedding_dim=config.embedding_dim
        )
        
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                mlp_size=config.mlp_size,
                attn_dropout=config.attn_dropout,
                mlp_dropout=config.mlp_dropout
            ) for _ in range(config.num_layers)
        ])
        
        self.classifier = MultiHeadPerceptronClassifier(
            embedding_dim=config.embedding_dim,
            num_classes=config.num_classes
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        x = self.patch_embedding(x)
        
        for encoder in self.transformer_encoders:
            x = encoder(x)
        
        x = x[:, 0]
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
        
        return logits, loss