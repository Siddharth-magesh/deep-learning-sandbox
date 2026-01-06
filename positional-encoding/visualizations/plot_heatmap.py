import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_heatmap(pe: torch.Tensor, title: str = "Positional Encoding", save_path: str = None, figsize: tuple = (12, 8)):
    if isinstance(pe, torch.Tensor):
        pe_data = pe.detach().cpu().numpy()
    else:
        pe_data = np.array(pe)
    
    plt.figure(figsize=figsize)
    sns.heatmap(pe_data, cmap="viridis", cbar_kws={'label': 'Value'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Embedding Dimension", fontsize=12)
    plt.ylabel("Position", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
