import torch
from . import config
from .encodings import SinusoidalPE, RotaryPE, ALiBiPE, BinaryPE, IndexPE
from .visualizations import plot_heatmap, plot_frequency, plot_rotation, plot_comparison, plot_3d_encoding
from .utils import normalize_encoding

def demonstrate_sinusoidal():
    print("=" * 60)
    print("Sinusoidal Positional Encoding")
    print("=" * 60)
    
    pe_encoder = SinusoidalPE(d_model=config.D_MODEL, seq_len=config.SEQ_LEN)
    pe = pe_encoder()
    
    print(f"Shape: {pe.shape}")
    print(f"Min value: {pe.min().item():.4f}")
    print(f"Max value: {pe.max().item():.4f}")
    print(f"Mean: {pe.mean().item():.4f}")
    print()
    
    plot_heatmap(pe, "Sinusoidal Positional Encoding")
    plot_frequency(pe, dims=[0, 2, 4, 8, 16, 32], title="Sinusoidal PE Frequencies")

def demonstrate_rotary():
    print("=" * 60)
    print("Rotary Positional Encoding")
    print("=" * 60)
    
    rope_encoder = RotaryPE(d_model=config.D_MODEL, max_seq_len=config.MAX_SEQ_LEN)
    
    x = torch.randn(config.SEQ_LEN, config.D_MODEL)
    positions = torch.arange(config.SEQ_LEN, dtype=torch.float32)
    
    rotated = rope_encoder(x, positions)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {rotated.shape}")
    print(f"Min value: {rotated.min().item():.4f}")
    print(f"Max value: {rotated.max().item():.4f}")
    print()
    
    plot_heatmap(rotated, "Rotary Positional Encoding Applied")
    plot_rotation(d_model=config.D_MODEL, num_positions=100)

def demonstrate_alibi():
    print("=" * 60)
    print("ALiBi Positional Encoding")
    print("=" * 60)
    
    alibi_encoder = ALiBiPE(num_heads=config.NUM_HEADS, max_seq_len=config.MAX_SEQ_LEN)
    bias = alibi_encoder(seq_len=config.SEQ_LEN)
    
    print(f"Shape: {bias.shape}")
    print(f"Number of heads: {config.NUM_HEADS}")
    print(f"Min value: {bias.min().item():.4f}")
    print(f"Max value: {bias.max().item():.4f}")
    print()
    
    for head_idx in range(min(3, config.NUM_HEADS)):
        plot_heatmap(bias[head_idx], f"ALiBi Head {head_idx} Attention Bias")

def demonstrate_binary():
    print("=" * 60)
    print("Binary Positional Encoding")
    print("=" * 60)
    
    binary_encoder = BinaryPE(seq_len=config.SEQ_LEN)
    pe = binary_encoder()
    
    print(f"Shape: {pe.shape}")
    print(f"Number of bits: {binary_encoder.bits}")
    print(f"Unique values: {pe.unique().tolist()}")
    print()
    
    plot_heatmap(pe, "Binary Positional Encoding")

def demonstrate_index():
    print("=" * 60)
    print("Index Positional Encoding")
    print("=" * 60)
    
    index_encoder = IndexPE(seq_len=config.SEQ_LEN, normalize=False)
    pe_raw = index_encoder()
    
    index_encoder_norm = IndexPE(seq_len=config.SEQ_LEN, normalize=True)
    pe_norm = index_encoder_norm()
    
    print(f"Raw shape: {pe_raw.shape}")
    print(f"Normalized shape: {pe_norm.shape}")
    print(f"Raw range: [{pe_raw.min().item():.4f}, {pe_raw.max().item():.4f}]")
    print(f"Normalized range: [{pe_norm.min().item():.4f}, {pe_norm.max().item():.4f}]")
    print()

def demonstrate_comparison():
    print("=" * 60)
    print("Comparing All Encodings")
    print("=" * 60)
    
    sinusoidal = SinusoidalPE(d_model=config.D_MODEL, seq_len=config.SEQ_LEN)()
    binary = BinaryPE(seq_len=config.SEQ_LEN)()
    
    rope_encoder = RotaryPE(d_model=config.D_MODEL)
    x = torch.randn(config.SEQ_LEN, config.D_MODEL)
    positions = torch.arange(config.SEQ_LEN, dtype=torch.float32)
    rotary = rope_encoder(x, positions)
    
    alibi_encoder = ALiBiPE(num_heads=config.NUM_HEADS)
    alibi = alibi_encoder(seq_len=config.SEQ_LEN)[0]
    
    encodings_dict = {
        "Sinusoidal": sinusoidal,
        "Rotary (Applied)": rotary,
        "Binary": binary,
        "ALiBi (Head 0)": alibi,
    }
    
    plot_comparison(encodings_dict)

def demonstrate_3d():
    print("=" * 60)
    print("3D Visualization")
    print("=" * 60)
    
    pe = SinusoidalPE(d_model=config.D_MODEL, seq_len=config.SEQ_LEN)()
    plot_3d_encoding(pe, "Sinusoidal PE - 3D View")

def main():
    print("\n" + "=" * 60)
    print("POSITIONAL ENCODING DEMONSTRATIONS")
    print("=" * 60 + "\n")
    
    demonstrate_sinusoidal()
    demonstrate_rotary()
    demonstrate_alibi()
    demonstrate_binary()
    demonstrate_index()
    demonstrate_comparison()
    demonstrate_3d()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

