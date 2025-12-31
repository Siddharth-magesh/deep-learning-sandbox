import torch

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, f"Height and Width must be divisible by window_size, got H: {H}, W: {W}, window_size: {window_size}"
    h_windows = H // window_size
    w_windows = W // window_size
    x = x.view(
        B,
        h_windows,
        window_size,
        w_windows,
        window_size,
        C
    ) # (B, H, W, C) -> (B, h_windows, window_size, w_windows, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # (B, h_windows, w_windows, window_size, window_size, C)
    windows = x.view(
        B * h_windows * w_windows,
        window_size,
        window_size,
        C
    ) # (B * h_windows * w_windows, window_size, window_size, C)
    return windows

def window_reverse(
        window: torch.Tensor,
        window_size: int,
        H: int,
        W: int
) -> torch.Tensor:
    assert H % window_size == 0 and W % window_size == 0, f"Height and Width must be divisible by window_size, got H: {H}, W: {W}, window_size: {window_size}"
    h_windows = H // window_size
    w_windows = W // window_size
    B = window.shape[0] // (h_windows * w_windows) 
    _, _, _, C = window.shape
    x = window.view(
        B,
        h_windows,
        w_windows,
        window_size,
        window_size,
        C
    ) # (B * h_windows * w_windows, window_size, window_size, C) -> (B, h_windows, w_windows, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # (B, h_windows, window_size, w_windows, window_size, C)
    x = x.view(B, H, W, C) # (B, H, W, C)
    return x

if __name__ == "__main__":
    B, H, W, C = 2, 56, 56, 96
    window_size = 7

    x = torch.randn(B, H, W, C)

    windows = window_partition(x, window_size)
    x_reconstructed = window_reverse(windows, window_size, H, W)

    print("Windows shape:", windows.shape)  # (2*64, 7, 7, 96)
    print("Reconstruction error:",
          (x - x_reconstructed).abs().max().item())

    assert torch.allclose(x, x_reconstructed), "Window ops are broken!"
