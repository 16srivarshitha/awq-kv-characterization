"""
Hadamard Transform (FWHT-128) Utilities for Post-RoPE Activation Smoothing.
Constructs normalized orthogonal 128x128 Hadamard matrix in pure PyTorch.
"""

import math
import torch


def get_hadamard_matrix_128(device="cuda", dtype=torch.float16) -> torch.Tensor:
    """
    Constructs an orthonormal Sylvester-Hadamard matrix of size 128x128:
      H_1 = [1]
      H_{2k} = H_2 \otimes H_k
      H_norm = H / sqrt(128)
    Satisfies H_norm^T @ H_norm = I.
    """
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32)
    for _ in range(6):  # 2^7 = 128
        H = torch.kron(H, torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32))
    
    H_norm = H / math.sqrt(128.0)
    return H_norm.to(dtype)


def apply_hadamard_inplace(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    Applies H along the last dimension: x_rot = x @ H.
    x: [..., 128]
    """
    return torch.matmul(x, H)
