from dataclasses import dataclass
from typing import Optional, Tuple

import logging
import torch


@dataclass
class SplatGatherMap1D:
    g0: torch.Tensor     # (N_s,) long
    g1: torch.Tensor     # (N_s,) long
    w0: torch.Tensor     # (N_s,) float
    w1: torch.Tensor     # (N_s,) float
    mask0: torch.Tensor  # (N_s,) bool
    mask1: torch.Tensor  # (N_s,) bool


def build_sg_map(points: torch.Tensor, a: float, b: float, s: int) -> SplatGatherMap1D:
    """
    Precompute indices and weights for linear splat/gather on a 1D uniform grid.
    points: (N_s,) tensor of coordinates (same device as returned tensors)
    Grid nodes: x_g = a + g * Δ, g=0..s-1, Δ = (b-a)/(s-1)
    """
    device = points.device
    N = points.numel()
    if s < 2:
        raise ValueError("Grid size s must be >= 2")
    a_t = torch.as_tensor(a, device=device, dtype=points.dtype)
    b_t = torch.as_tensor(b, device=device, dtype=points.dtype)
    delta = (b_t - a_t) / (s - 1)
    # Protect against degenerate interval
    delta = torch.clamp(delta, min=torch.finfo(points.dtype).eps)

    rel = (points - a_t) / delta  # (N,)
    g0 = torch.floor(rel).to(torch.long)
    g0 = torch.clamp(g0, 0, s - 2)
    g1 = g0 + 1

    xg0 = a_t + g0.to(points.dtype) * delta
    w1 = (points - xg0) / delta
    w1 = torch.clamp(w1, 0.0, 1.0)
    w0 = 1.0 - w1

    # Out-of-bounds masks (strictly below a or above b)
    mask0 = points >= a_t
    mask1 = points <= b_t
    mask = mask0 & mask1
    # For safety, if not mask, set weights to zero
    w0 = w0 * mask
    w1 = w1 * mask

    return SplatGatherMap1D(
        g0=g0, g1=g1, w0=w0.to(points.dtype), w1=w1.to(points.dtype),
        mask0=mask, mask1=mask
    )


def _nearest_fill_empty(grid: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Fill empty positions (valid==False) along last dimension by nearest neighbor
    from valid cells. grid: (B,C,S) or (C,S). valid: (S,) or (B,S).
    """
    if grid.dim() == 2:
        grid = grid.unsqueeze(0)  # (1,C,S)
        squeeze_back = True
    else:
        squeeze_back = False

    B, C, S = grid.shape
    if valid.dim() == 1:
        valid = valid.unsqueeze(0).expand(B, S)  # (B,S)

    # Forward fill
    idx = torch.arange(S, device=grid.device).unsqueeze(0).expand(B, S)
    # Replace invalid positions with previous valid index
    last_valid = torch.zeros(B, dtype=torch.long, device=grid.device)
    out = grid.clone()
    for s in range(S):
        take = valid[:, s]
        last_valid = torch.where(take, idx[:, s], last_valid)
        out[:, :, s] = grid[torch.arange(B, device=grid.device), :, last_valid]

    # Backward fill to handle leading empties
    last_valid = torch.full((B,), S - 1, dtype=torch.long, device=grid.device)
    for s in reversed(range(S)):
        take = valid[:, s]
        last_valid = torch.where(take, idx[:, s], last_valid)
        out[:, :, s] = out[torch.arange(B, device=grid.device), :, last_valid]

    if squeeze_back:
        out = out.squeeze(0)
    return out


def points_to_grid(values: torch.Tensor, sg: SplatGatherMap1D, s: int, eps: float = 1e-12,
                   nearest_fill: bool = True, warn_threshold: float = 0.2) -> torch.Tensor:
    """
    Linear splat from points to grid.
    values: (B,C,N_s) or (C,N_s)
    Returns: grid (B,C,s) or (C,s)
    """
    if values.dim() == 2:
        values = values.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    B, C, N = values.shape
    device = values.device

    num = torch.zeros(B, C, s, device=device, dtype=values.dtype)
    den = torch.zeros(B, 1, s, device=device, dtype=values.dtype)

    # Scatter add for g0 and g1
    for g, w in ((sg.g0, sg.w0), (sg.g1, sg.w1)):
        g_exp = g.view(1, 1, N).expand(B, C, N)
        w_exp = w.view(1, 1, N).expand(B, C, N)
        num.scatter_add_(dim=2, index=g_exp, src=values * w_exp)
        den.scatter_add_(dim=2, index=g.view(1, 1, N).expand(B, 1, N), src=w.view(1, 1, N).expand(B, 1, N))

    grid = num / (den + eps)

    # Nearest neighbor fill for empty cells
    empty = (den.squeeze(1) <= eps)
    empty_ratio = empty.float().mean().item()
    if empty_ratio > warn_threshold:
        logging.warning(f"points_to_grid: {empty_ratio*100:.1f}% empty cells; consider increasing s_min or reducing s_total")
    if nearest_fill:
        grid = _nearest_fill_empty(grid, ~empty)

    if squeeze_back:
        grid = grid.squeeze(0)
    return grid


def grid_to_points(grid: torch.Tensor, sg: SplatGatherMap1D) -> torch.Tensor:
    """
    Linear gather from grid to points.
    grid: (B,C,S) or (C,S)
    Returns: values_hat (B,C,N) or (C,N)
    """
    if grid.dim() == 2:
        grid = grid.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False
    B, C, S = grid.shape
    N = sg.g0.numel()

    g0 = sg.g0.view(1, 1, N).expand(B, C, N)
    g1 = sg.g1.view(1, 1, N).expand(B, C, N)
    w0 = sg.w0.view(1, 1, N).expand(B, C, N)
    w1 = sg.w1.view(1, 1, N).expand(B, C, N)

    vals = w0 * torch.gather(grid, dim=2, index=g0) + w1 * torch.gather(grid, dim=2, index=g1)

    if squeeze_back:
        vals = vals.squeeze(0)
    return vals

