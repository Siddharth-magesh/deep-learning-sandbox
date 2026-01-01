import torch
from modules import RelativePositionBias
from experiments import visualize_relative_position_bias, visualize_bias_table
from pathlib import Path


def demo_1d_bias():
    print("=" * 50)
    print("1D Relative Position Bias Demo")
    print("=" * 50)
    
    seq_len = 16
    num_heads = 4
    
    rpb_1d = RelativePositionBias(
        num_heads=num_heads,
        seq_len=seq_len,
        bias_type="1d",
        init_std=0.02
    )
    
    bias = rpb_1d()
    print(f"Bias shape: {bias.shape}")
    print(f"Bias table shape: {rpb_1d.relative_position_bias_table.shape}")
    print(f"Number of relative positions: {rpb_1d.num_relative_positions}")
    
    Path("visualizations").mkdir(exist_ok=True)
    visualize_relative_position_bias(
        bias,
        save_path="visualizations/1d_bias.png",
        title="1D Relative Position Bias (First Head)"
    )
    visualize_bias_table(
        rpb_1d.relative_position_bias_table,
        save_path="visualizations/1d_bias_table.png",
        title="1D Bias Table (All Heads)"
    )
    print("Visualizations saved!\n")


def demo_2d_bias():
    print("=" * 50)
    print("2D Relative Position Bias Demo")
    print("=" * 50)
    
    window_size = (7, 7)
    num_heads = 8
    
    rpb_2d = RelativePositionBias(
        num_heads=num_heads,
        window_size=window_size,
        bias_type="2d",
        init_std=0.02
    )
    
    bias = rpb_2d()
    print(f"Window size: {window_size}")
    print(f"Bias shape: {bias.shape}")
    print(f"Bias table shape: {rpb_2d.relative_position_bias_table.shape}")
    print(f"Number of relative positions: {rpb_2d.num_relative_positions}")
    
    visualize_relative_position_bias(
        bias,
        save_path="visualizations/2d_bias.png",
        title="2D Relative Position Bias (First Head)",
        cmap="coolwarm"
    )
    visualize_bias_table(
        rpb_2d.relative_position_bias_table,
        save_path="visualizations/2d_bias_table.png",
        title="2D Bias Table (All Heads)"
    )
    print("Visualizations saved!\n")


def demo_comparison():
    print("=" * 50)
    print("Comparison Demo")
    print("=" * 50)
    
    seq_len = 49
    window_size = (7, 7)
    num_heads = 4
    
    rpb_1d = RelativePositionBias(
        num_heads=num_heads,
        seq_len=seq_len,
        bias_type="1d",
        init_std=0.02
    )
    
    rpb_2d = RelativePositionBias(
        num_heads=num_heads,
        window_size=window_size,
        bias_type="2d",
        init_std=0.02
    )
    
    bias_1d = rpb_1d()
    bias_2d = rpb_2d()
    
    print(f"1D - Number of parameters: {rpb_1d.num_relative_positions * num_heads}")
    print(f"2D - Number of parameters: {rpb_2d.num_relative_positions * num_heads}")
    print(f"Sequence length: {seq_len}")
    print(f"Window size: {window_size}")
    print()


if __name__ == "__main__":
    demo_1d_bias()
    demo_2d_bias()
    demo_comparison()
    print("All demos completed successfully!")
