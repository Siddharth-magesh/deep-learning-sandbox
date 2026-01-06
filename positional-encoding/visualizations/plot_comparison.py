import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_comparison(encodings_dict: dict, save_path: str = None, figsize: tuple = (15, 10)):
    num_encodings = len(encodings_dict)
    cols = 2
    rows = (num_encodings + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_encodings == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, encoding) in enumerate(encodings_dict.items()):
        if isinstance(encoding, torch.Tensor):
            data = encoding.detach().cpu().numpy()
        else:
            data = np.array(encoding)
        
        im = axes[idx].imshow(data, cmap='viridis', aspect='auto')
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Dimension', fontsize=10)
        axes[idx].set_ylabel('Position', fontsize=10)
        plt.colorbar(im, ax=axes[idx])
    
    for idx in range(num_encodings, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def plot_3d_encoding(pe: torch.Tensor, title: str = "3D Positional Encoding", save_path: str = None):
    from mpl_toolkits.mplot3d import Axes3D
    
    if isinstance(pe, torch.Tensor):
        pe_data = pe.detach().cpu().numpy()
    else:
        pe_data = np.array(pe)
    
    if pe_data.shape[1] < 3:
        print("Need at least 3 dimensions for 3D plot")
        return
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = pe_data[:, 0]
    y = pe_data[:, 1]
    z = pe_data[:, 2]
    
    positions = np.arange(len(pe_data))
    scatter = ax.scatter(x, y, z, c=positions, cmap='viridis', s=50, alpha=0.6)
    
    ax.set_xlabel('Dimension 0', fontsize=12)
    ax.set_ylabel('Dimension 1', fontsize=12)
    ax.set_zlabel('Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, label='Position')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
