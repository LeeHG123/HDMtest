# functions/kernels.py
# Spectral Mixture (SM) & Windowed SM (WSM) kernels for HilbertNoise
# - 모든 커널 조립은 float64에서 수행 (수치안정)
# - 반환은 "순수 커널값" (대칭화/지터는 호출자가 수행)

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import Tuple

# ───────────── 공통 유틸 ─────────────

def normalize_coords(x: torch.Tensor, offset: float, scale: float) -> torch.Tensor:
    """
    x: (..., N)  실좌표
    return: (..., N)  \tilde x = (x - offset) / scale, dtype=float64
    """
    xt = (x - float(offset)) / float(scale)
    return xt.to(torch.float64)

def pairwise_delta(x1_tilde: torch.Tensor, x2_tilde: torch.Tensor) -> torch.Tensor:
    """
    x1_tilde: (N1,), x2_tilde: (N2,)  — float64
    return: (N1, N2) with Δ = x1 - x2
    """
    if x1_tilde.dim() != 1 or x2_tilde.dim() != 1:
        raise ValueError("x1_tilde, x2_tilde must be 1D.")
    return (x1_tilde[:, None] - x2_tilde[None, :]).to(torch.float64)

# ───────────── 창(윈도우) ─────────────

def gaussian_window(x_tilde: torch.Tensor,
                    c: torch.Tensor,
                    omega_raw: torch.Tensor,
                    eps_w: float = 1e-6) -> torch.Tensor:
    """
    x_tilde: (N,)
    c: (Q,)          — 중심 (정규화 좌표계)
    omega_raw: (Q,)  — softplus 로 양수 보장
    return: s(x) (Q, N) ≥ 0  in float64
    """
    x_tilde = x_tilde.to(torch.float64)
    c = c.to(torch.float64).view(-1)              # (Q,)
    omega = F.softplus(omega_raw.to(torch.float64).view(-1)) + float(eps_w)
    # (Q,1) vs (1,N) → (Q,N)
    num = (x_tilde[None, :] - c[:, None]) ** 2
    den = 2.0 * (omega[:, None] ** 2)
    s = torch.exp(- num / den)
    return s  # (Q, N)

# ───────────── 정상성 SM 커널 ─────────────

def kernel_sm(x1_tilde: torch.Tensor,
              x2_tilde: torch.Tensor,
              w_raw: torch.Tensor,
              mu: torch.Tensor,
              sigma_raw: torch.Tensor,
              eps_w: float = 1e-6,
              eps_sigma: float = 1e-6) -> torch.Tensor:
    r"""
    k_SM(Δ) = \sum_q w_q * exp( - 2 π^2 σ_q^2 Δ^2 ) * cos( 2 π μ_q Δ )
    입력은 모두 float64로 처리.
    """
    x1_tilde = x1_tilde.to(torch.float64).view(-1)
    x2_tilde = x2_tilde.to(torch.float64).view(-1)
    Δ = pairwise_delta(x1_tilde, x2_tilde)                     # (N1, N2)

    w = F.softplus(w_raw.to(torch.float64).view(-1)) + float(eps_w)      # (Q,)
    mu = mu.to(torch.float64).view(-1)                                   # (Q,)
    sigma = F.softplus(sigma_raw.to(torch.float64).view(-1)) + float(eps_sigma)

    # (Q, N1, N2)
    Δ2 = Δ[None, :, :] ** 2
    two_pi = 2.0 * math.pi
    expo = torch.exp(- (two_pi ** 2) * (sigma[:, None, None] ** 2) * Δ2)
    osc  = torch.cos(two_pi * mu[:, None, None] * Δ)
    K_q = (w[:, None, None] * expo * osc)                                # (Q,N1,N2)
    K = K_q.sum(dim=0)                                                   # (N1,N2) float64
    return K

# ───────────── 비정상성 WSM 커널 ─────────────

def kernel_wsm(x1_tilde: torch.Tensor,
               x2_tilde: torch.Tensor,
               w_raw: torch.Tensor,
               mu: torch.Tensor,
               sigma_raw: torch.Tensor,
               c: torch.Tensor,
               omega_raw: torch.Tensor,
               eps_w: float = 1e-6,
               eps_sigma: float = 1e-6) -> torch.Tensor:
    r"""
    k_WSM(x,x') = \sum_q s_q(x) k_SM,q(x - x') s_q(x')
    여기서는 k_SM,q = w_q * exp(-2π^2 σ_q^2 Δ^2) * cos(2π μ_q Δ)
    """
    # 창값 (Q,N1), (Q,N2)
    s1 = gaussian_window(x1_tilde, c, omega_raw, eps_w=eps_w)  # (Q,N1)
    s2 = gaussian_window(x2_tilde, c, omega_raw, eps_w=eps_w)  # (Q,N2)

    # Δ는 한 번만
    x1_tilde = x1_tilde.to(torch.float64).view(-1)
    x2_tilde = x2_tilde.to(torch.float64).view(-1)
    Δ = pairwise_delta(x1_tilde, x2_tilde)                                 # (N1,N2)

    w = F.softplus(w_raw.to(torch.float64).view(-1)) + float(eps_w)        # (Q,)
    mu = mu.to(torch.float64).view(-1)                                      # (Q,)
    sigma = F.softplus(sigma_raw.to(torch.float64).view(-1)) + float(eps_sigma)

    two_pi = 2.0 * math.pi
    Δ2 = Δ[None, :, :] ** 2                                                 # (1,N1,N2)
    expo = torch.exp(- (two_pi ** 2) * (sigma[:, None, None] ** 2) * Δ2)    # (Q,N1,N2)
    osc  = torch.cos(two_pi * mu[:, None, None] * Δ)                        # (Q,N1,N2)
    Kq   = (w[:, None, None] * expo * osc)                                  # (Q,N1,N2)

    # 창 대칭 곱: (Q,N1) * (Q,N2)
    K = (s1[:, :, None] * Kq * s2[:, None, :]).sum(dim=0)                   # (N1,N2)
    return K.to(torch.float64)
