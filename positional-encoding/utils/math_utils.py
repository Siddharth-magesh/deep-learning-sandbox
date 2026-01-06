import torch
import math

def compute_frequencies(d_model: int, base: float = 10000.0) -> torch.Tensor:
    half_dim = d_model // 2
    freqs = torch.exp(
        -math.log(base) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    )
    return freqs

def apply_rotary_embedding(x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
    sin_val = torch.sin(angles)
    cos_val = torch.cos(angles)
    
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    
    rotated = torch.cat([
        x1 * cos_val - x2 * sin_val,
        x1 * sin_val + x2 * cos_val
    ], dim=-1)
    
    return rotated

def relative_position_matrix(seq_len: int) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float32)
    relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
    return relative_positions

def gaussian_kernel(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    return torch.exp(-0.5 * (x / sigma) ** 2)

def normalize_encoding(pe: torch.Tensor, method: str = 'l2') -> torch.Tensor:
    if method == 'l2':
        norm = torch.norm(pe, p=2, dim=-1, keepdim=True)
        return pe / (norm + 1e-8)
    elif method == 'min_max':
        min_val = pe.min(dim=-1, keepdim=True)[0]
        max_val = pe.max(dim=-1, keepdim=True)[0]
        return (pe - min_val) / (max_val - min_val + 1e-8)
    elif method == 'standard':
        mean = pe.mean(dim=-1, keepdim=True)
        std = pe.std(dim=-1, keepdim=True)
        return (pe - mean) / (std + 1e-8)
    else:
        return pe

def get_attention_mask(seq_len: int, causal: bool = False) -> torch.Tensor:
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    else:
        return torch.zeros(seq_len, seq_len).bool()

def interpolate_encoding(pe: torch.Tensor, target_len: int) -> torch.Tensor:
    if pe.dim() == 2:
        pe = pe.unsqueeze(0)
    
    original_len = pe.shape[1]
    if original_len == target_len:
        return pe.squeeze(0)
    
    pe_transposed = pe.transpose(1, 2)
    interpolated = torch.nn.functional.interpolate(
        pe_transposed, 
        size=target_len, 
        mode='linear', 
        align_corners=True
    )
    interpolated = interpolated.transpose(1, 2)
    
    return interpolated.squeeze(0)
