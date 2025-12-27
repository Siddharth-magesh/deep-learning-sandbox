import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, use_bias: bool):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.query_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.key_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.value_projection = nn.Linear(d_model, d_model, bias=use_bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=use_bias)
        
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = self.query_projection(hidden_states)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)
        
        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        
        output = self.output_projection(context)
        output = self.residual_dropout(output)
        
        return output
