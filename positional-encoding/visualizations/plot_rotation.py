import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_rotation(d_model: int = 64, num_positions: int = 100, save_path: str = None, figsize: tuple = (10, 10)):
    angles = np.linspace(0, 2 * np.pi, num_positions)
    x = np.cos(angles)
    y = np.sin(angles)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    ax1.plot(x, y, linewidth=2, alpha=0.7)
    ax1.scatter(x[0], y[0], color="red", s=100, label="Start", zorder=5)
    ax1.scatter(x[-1], y[-1], color="blue", s=100, label="End", zorder=5)
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Rotary Encoding Trajectory", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Real Component", fontsize=12)
    ax1.set_ylabel("Imaginary Component", fontsize=12)
    ax1.legend()
    
    frequencies = torch.exp(
        -np.log(10000.0) * torch.arange(0, d_model // 2, dtype=torch.float32) / (d_model // 2)
    )
    positions = torch.arange(num_positions, dtype=torch.float32)
    angles_matrix = positions.unsqueeze(-1) * frequencies.unsqueeze(0)
    
    ax2.plot(frequencies.numpy(), marker='o', linestyle='-', linewidth=2)
    ax2.set_title("Frequency Distribution", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Dimension Index", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
