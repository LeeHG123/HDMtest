# evaluate/moments.py
# 1) Mean-function error (L2)  2) Covariance-operator error (Hilbert–Schmidt)

from __future__ import annotations
from typing import Tuple, Optional
import torch


def _as_2d_y(y: torch.Tensor) -> torch.Tensor:
    """
    Accepts (B,N), (B,N,1), (N,), (N,1) and returns (B,N).
    """
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    y = y.detach()
    if y.dim() == 1:
        y = y.unsqueeze(0)
    if y.dim() == 2:
        return y
    if y.dim() == 3 and y.size(-1) == 1:
        return y.squeeze(-1)
    raise ValueError(f"Unsupported y shape: {tuple(y.shape)}")


def _as_1d_x(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts (N), (N,1), (B,N), (B,N,1) and returns (N).
    If batched, uses the first row (assumes common evaluation grid).
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach()
    if x.dim() == 1:
        return x
    if x.dim() == 2:
        # (N,1) or (B,N)
        if x.size(-1) == 1:  # (N,1)
            return x.squeeze(-1)
        return x[0]
    if x.dim() == 3 and x.size(-1) == 1:
        # (B,N,1)
        return x[0].squeeze(-1)
    raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")


def trapz_weights_1d(x_1d: torch.Tensor) -> torch.Tensor:
    """
    Trapezoidal quadrature weights for 1D grid x (assumed sorted ascending).
    Returns w of shape (N,) such that sum_i w_i f(x_i) approximates ∫ f(x) dx.
    """
    x = _as_1d_x(x_1d).double().cpu().flatten()
    n = x.numel()
    if n < 2:
        return torch.ones_like(x)

    dx = x[1:] - x[:-1]
    w = torch.empty(n, dtype=torch.double)
    w[0] = dx[0] / 2.0
    w[-1] = dx[-1] / 2.0
    if n > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    return w


def mean_function_error(
    x: torch.Tensor,
    y_gen: torch.Tensor,
    y_data: torch.Tensor,
) -> float:
    """
    L2(Ω) norm of mean-function mismatch:
      || m_gen - m_data ||_{L2}  ≈ sqrt( Σ_i w_i (m_gen(x_i)-m_data(x_i))^2 ).
    """
    x1d = _as_1d_x(x)
    w = trapz_weights_1d(x1d)  # (N,)

    Yg = _as_2d_y(y_gen).double().cpu()
    Yd = _as_2d_y(y_data).double().cpu()

    if Yg.shape[1] != w.numel() or Yd.shape[1] != w.numel():
        raise ValueError(
            f"Grid length mismatch: len(x)={w.numel()}, y_gen={tuple(Yg.shape)}, y_data={tuple(Yd.shape)}"
        )

    m_g = Yg.mean(dim=0)  # (N,)
    m_d = Yd.mean(dim=0)  # (N,)
    diff = m_g - m_d
    err2 = (w * diff * diff).sum()
    return float(torch.sqrt(err2))


def covariance_operator_error(
    x: torch.Tensor,
    y_gen: torch.Tensor,
    y_data: torch.Tensor,
    unbiased: bool = True,
) -> float:
    """
    Hilbert–Schmidt norm of covariance-operator mismatch:
      || C_gen - C_data ||_{HS}^2 = ∬ (k_gen(x,y)-k_data(x,y))^2 dx dy
    Discrete approx:
      ≈ Σ_i Σ_j w_i w_j (K_gen[i,j] - K_data[i,j])^2,
    where K is the sample covariance matrix of function values on the grid.
    """
    x1d = _as_1d_x(x)
    w = trapz_weights_1d(x1d)  # (N,)

    Yg = _as_2d_y(y_gen).double().cpu()  # (B,N)
    Yd = _as_2d_y(y_data).double().cpu()

    N = w.numel()
    if Yg.shape[1] != N or Yd.shape[1] != N:
        raise ValueError(
            f"Grid length mismatch: len(x)={N}, y_gen={tuple(Yg.shape)}, y_data={tuple(Yd.shape)}"
        )

    Bg = Yg.shape[0]
    Bd = Yd.shape[0]

    mg = Yg.mean(dim=0, keepdim=True)
    md = Yd.mean(dim=0, keepdim=True)
    Xg = Yg - mg
    Xd = Yd - md

    denom_g = (Bg - 1) if (unbiased and Bg > 1) else max(Bg, 1)
    denom_d = (Bd - 1) if (unbiased and Bd > 1) else max(Bd, 1)

    Kg = (Xg.t() @ Xg) / float(denom_g)  # (N,N)
    Kd = (Xd.t() @ Xd) / float(denom_d)

    D = Kg - Kd
    ww = w[:, None] * w[None, :]  # (N,N)
    err2 = (D * D * ww).sum()
    return float(torch.sqrt(err2))


def moments_metrics(
    x: torch.Tensor,
    y_gen: torch.Tensor,
    y_data: torch.Tensor,
) -> Tuple[float, float]:
    """
    Convenience wrapper returning (mean_L2_error, cov_HS_error).
    """
    m_err = mean_function_error(x, y_gen, y_data)
    c_err = covariance_operator_error(x, y_gen, y_data)
    return m_err, c_err
