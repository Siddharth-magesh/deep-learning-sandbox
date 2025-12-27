import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, 
                 attention_dropout: float, layer_norm_epsilon: float, use_bias: bool):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.attention = MultiHeadAttention(d_model, num_heads, attention_dropout, use_bias)
        
        self.layer_norm2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.feedforward = FeedForward(d_model, d_ff, dropout, use_bias)
        
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.feedforward(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states
