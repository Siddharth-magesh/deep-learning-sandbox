import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def visualize_attention_weights(
    attn_weights: torch.Tensor,
    save_path: str | Path | None = None,
    title: str = "Attention Weights",
    cmap: str = "viridis"
) -> None:
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0, 0]
    elif attn_weights.dim() == 3:
        attn_weights = attn_weights[0]
        
    attn_weights = attn_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap=cmap, square=True, cbar=True)
    plt.title(title)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def visualize_relative_position_bias(
    bias: torch.Tensor,
    save_path: str | Path | None = None,
    title: str = "Relative Position Bias",
    cmap: str = "coolwarm"
) -> None:
    if bias.dim() == 3:
        bias = bias[0]
        
    bias = bias.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(bias, cmap=cmap, center=0, square=True, cbar=True)
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def visualize_bias_table(
    bias_table: torch.Tensor,
    save_path: str | Path | None = None,
    title: str = "Bias Table"
) -> None:
    bias_table = bias_table.detach().cpu().numpy()
    num_positions, num_heads = bias_table.shape
    
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(bias_table[:, i])
        ax.set_title(f"Head {i}")
        ax.set_xlabel("Relative Position Index")
        ax.set_ylabel("Bias Value")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
