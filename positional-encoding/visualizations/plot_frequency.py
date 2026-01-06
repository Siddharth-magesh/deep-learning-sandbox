import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_frequency(pe: torch.Tensor, dims: list = None, title: str = "Frequency Components", save_path: str = None, figsize: tuple = (12, 6)):
    if isinstance(pe, torch.Tensor):
        pe_data = pe.detach().cpu().numpy()
    else:
        pe_data = np.array(pe)
    
    if dims is None:
        total_dims = pe_data.shape[1]
        if total_dims <= 10:
            dims = list(range(total_dims))
        else:
            dims = [0, 2, 4, total_dims // 4, total_dims // 2, total_dims - 1]
    
    plt.figure(figsize=figsize)
    for dim in dims:
        if dim < pe_data.shape[1]:
            plt.plot(pe_data[:, dim], label=f"Dimension {dim}", alpha=0.8)
    
    plt.legend(loc='best', fontsize=10)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
