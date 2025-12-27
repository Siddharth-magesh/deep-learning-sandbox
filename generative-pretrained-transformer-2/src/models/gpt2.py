import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from ..config import GPT2Config
from ..modules import DecoderBlock


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.residual_dropout,
                attention_dropout=config.attention_dropout,
                layer_norm_epsilon=config.layer_norm_epsilon,
                use_bias=config.use_bias
            ) for _ in range(config.num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_causal_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length = input_ids.shape
        device = input_ids.device
        
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)
        
        attention_mask = self.get_causal_mask(sequence_length, device)
        
        for decoder_block in self.decoder_blocks:
            hidden_states = decoder_block(hidden_states, attention_mask)
        
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 repetition_penalty: float = 1.0, do_sample: bool = True):
        self.eval()
        
        for _ in range(max_new_tokens):
            input_ids_cond = input_ids if input_ids.size(1) <= self.config.context_length else input_ids[:, -self.config.context_length:]
            
            logits, _ = self.forward(input_ids_cond)
            logits = logits[:, -1, :] / temperature
            
            if repetition_penalty != 1.0:
                for batch_idx in range(input_ids.shape[0]):
                    for token_id in set(input_ids[batch_idx].tolist()):
                        logits[batch_idx, token_id] /= repetition_penalty
            
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_values[:, [-1]]] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            yield next_token
        
        return input_ids
