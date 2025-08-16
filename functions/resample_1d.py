from typing import Literal

import torch
import torch.nn.functional as F


def fft_resample_1d(x: torch.Tensor, S_target: int) -> torch.Tensor:
    """
    Resample real 1D signals along the last dimension to length S_target using
    rFFT -> crop/zero-pad -> iFFT. Shapes: x: (B,C,S_src) → (B,C,S_target).
    Handles even/odd Nyquist bin lengths.
    """
    if x.dim() != 3:
        raise ValueError("fft_resample_1d expects (B,C,S) input")
    B, C, S_src = x.shape
    if S_src == S_target:
        return x

    X = torch.fft.rfft(x, n=S_src, dim=2)  # (B,C,K_src)
    K_src = X.shape[2]
    K_tgt = S_target // 2 + 1

    # Allocate target spectrum
    X_tgt = torch.zeros(B, C, K_tgt, device=x.device, dtype=X.dtype)
    # Copy overlapping low-frequency part
    K_min = min(K_src, K_tgt)
    X_tgt[..., :K_min] = X[..., :K_min]

    # Special care for Nyquist when both even
    if (S_src % 2 == 0) and (S_target % 2 == 0) and (K_min == K_tgt):
        # Ensure Nyquist bin is real (imag part zero)
        X_tgt[..., -1] = X_tgt[..., -1].real

    y = torch.fft.irfft(X_tgt, n=S_target, dim=2)
    return y


def linear_resample_1d(x: torch.Tensor, S_target: int, align_corners: bool = True) -> torch.Tensor:
    """
    Linear interpolation along last dimension.
    x: (B,C,S_src) → (B,C,S_target)
    """
    if x.dim() != 3:
        raise ValueError("linear_resample_1d expects (B,C,S) input")
    if x.shape[-1] == S_target:
        return x
    return F.interpolate(x, size=S_target, mode='linear', align_corners=align_corners)

