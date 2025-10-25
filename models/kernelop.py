# -*- coding: utf-8 -*-
"""
kernelop.py — HDM backbone implementing the paper's operator-learning map  Ḡ = χ ∘ f̄ ∘ φ

This module replaces the existing kernelop.py with a faithful implementation of the
two-stage kernel operator learning pipeline described in
"Kernel methods are competitive for operator learning" (JCP 2024).

Mathematical mapping to the paper
---------------------------------
Let φ: U → R^n and ϕ=φ,  varphi=φ_out : V → R^m be the input/output linear measurement operators
given by fixed sensor sets X_in, X_out. For data {(u_i, v_i)} with U_i = φ(u_i) ∈ R^n and
V_i = varphi(v_i) ∈ R^m, define block-vectors 𝚄, 𝚅 of length N.

1) Input-space regression (vector-valued kernel with diagonal structure):
   Choose a scalar kernel S and set Γ(U,U') = S(U,U') I_m. Then
   f̄(U) = Γ(U,𝚄) ( Γ(𝚄,𝚄) + λ I )^{-1} 𝚅          (Eq. (2.12) with γ≡λ)
   For U = 𝚄 (evaluation at the "training/dictionary" points), predictions are
   ŷ_out^meas = S(𝚄,𝚄) α  where  α = ( S(𝚄,𝚄) + λI )^{-1} 𝚅      (Eq. (2.3), (2.12))

   This module supports both exact solves via Cholesky and a very efficient Nyström
   low-rank approximation using Woodbury identities for per-minibatch recomputation.

2) Output reconstruction (kernel interpolation in output RKHS):
   With a scalar kernel k_out on the output domain and fixed output sensors X_out,
   χ(y_•)(x) ≈ k_out(x, X_out) ( k_out(X_out, X_out) + ρ I )^{-1} y_•    (Eq. (2.13))

Hyper-parameters (λ, ρ, length-scales) are chosen by maximizing the (type-II) log marginal
likelihoods implied by the GP views in Section 2 (Eqs. (2.12)-(2.13)).

API highlights
--------------
- KernelOpHDM: single nn.Module you can drop in as a "non-trainable" HDM backbone.
  It exposes:
    * fit_output_kernel_hyperparams(Y)    # MLE for (ℓ_out, σ_out, ρ) from stacked outputs Y (N×m)
    * fit_input_kernel_hyperparams(U, V)  # MLE for (ℓ_in, σ_in, λ) from dictionary (U: N×d_in, V: N×m)
    * forward(U_batch, V_batch, x_query, t_batch=None)
         - recomputes α = (K+λI)^{-1} V_batch per *minibatch/time* exactly or via Nyström,
           then returns reconstructed fields at x_query via χ.

- Product-in-time kernel S((U,t), (U',t')) = S_in(U,U') * k_t(t,t')
  implemented by using anisotropic length-scales and concatenating t as the last column.

- Exact solver is numerically robust (Cholesky + jitter); Nyström solver can reduce
  O(N^3) → O(N M^2 + M^3) per batch with M≪N while remaining faithful to the formulation.

PyTorch-only, no autograd through the linear solves by default (backbone is "fixed").
If you want to learn sensors or embed this in a larger trainable network, you can
enable gradients by setting requires_grad on inputs/hyperparams explicitly.

Author: (you)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def _to_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x[:, None]
    return x

def _chol_with_jitter(A: torch.Tensor, init: float = 1e-12, max_tries: int = 8) -> torch.Tensor:
    """
    Numerically stable Cholesky: A is symmetrized, jitter is increased by 10x each time.
    Last resort: eigen-based PSD projection.
    """
    A = 0.5 * (A + A.T)  # symmetrize
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    jitter = float(init)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(A + jitter * I)
        except RuntimeError:
            jitter *= 10.0
    evals, evecs = torch.linalg.eigh(A)
    evals = evals.clamp_min(jitter)
    A_psd = (evecs * evals) @ evecs.T
    return torch.linalg.cholesky(A_psd + jitter * I)

def pairwise_sqdist(X: torch.Tensor, Y: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic squared distance: sum_j ((x_j - y_j)/ℓ_j)^2

    X: (N, d), Y: (M, d), lengthscale: (d,) positive
    Return: (N, M)
    """
    X = _to_2d(X)
    Y = _to_2d(Y)
    dev = lengthscale.device
    X = X.to(dev)
    Y = Y.to(dev)
    # Scale by lengthscale per dimension
    ls = lengthscale.view(1, -1)
    Xs = X / ls
    Ys = Y / ls
    # ||Xs||^2 + ||Ys||^2 - 2 Xs·Ys^T
    X2 = (Xs**2).sum(dim=1, keepdim=True)        # (N,1)
    Y2 = (Ys**2).sum(dim=1, keepdim=True).T      # (1,M)
    K = X2 + Y2 - 2.0 * (Xs @ Ys.T)
    # Numerical floor
    return torch.clamp(K, min=0.0)
# -----------------------------
# Scalar kernels S, k_out
# -----------------------------

class RBFKernel(nn.Module):
    """
    Squared Exponential (Gaussian) kernel:
        k(x,x') = σ^2 * exp( - ||x - x'||^2 / (2 ℓ^2) )
    with anisotropic ℓ (vector).
    """
    def __init__(self, lengthscale: torch.Tensor, variance: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.register_buffer('lengthscale', lengthscale.clone())
        self.register_buffer('variance', torch.tensor(float(variance)))
        self.eps = eps

    def set_params(self, lengthscale: torch.Tensor, variance: float):
        self.lengthscale = lengthscale.clone().to(self.lengthscale.device)
        self.variance = torch.tensor(float(variance), device=self.lengthscale.device)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        D2 = pairwise_sqdist(X, Y, self.lengthscale + self.eps)
        return self.variance * torch.exp(-0.5 * D2)


class RationalQuadraticKernel(nn.Module):
    """
    Rational Quadratic kernel (Appendix C):
        k(r) = σ^2 * (1 + r^2 / (2 α ℓ^2))^{-α}
    with anisotropic ℓ (vector), scalar α > 0.
    """
    def __init__(self, lengthscale: torch.Tensor, alpha: float = 1.0, variance: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.register_buffer('lengthscale', lengthscale.clone())
        self.alpha = float(alpha)
        self.register_buffer('variance', torch.tensor(float(variance)))
        self.eps = eps

    def set_params(self, lengthscale: torch.Tensor, alpha: float, variance: float):
        self.lengthscale = lengthscale.clone().to(self.lengthscale.device)
        self.alpha = float(alpha)
        self.variance = torch.tensor(float(variance), device=self.lengthscale.device)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        D2 = pairwise_sqdist(X, Y, self.lengthscale + self.eps)  # ~ sum_j ((x_j - y_j)/ℓ_j)^2
        base = 1.0 + 0.5 * D2 / max(self.alpha, self.eps)
        return self.variance * torch.pow(base, -self.alpha)


class Matern52Kernel(nn.Module):
    """
    Matérn ν=5/2 kernel with anisotropic ℓ (vector):
        k(r) = σ^2 * (1 + √5 r + 5 r^2/3) exp(-√5 r)
    where r = sqrt(sum_j ((x_j - y_j)/ℓ_j)^2 )
    """
    def __init__(self, lengthscale: torch.Tensor, variance: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.register_buffer('lengthscale', lengthscale.clone())
        self.register_buffer('variance', torch.tensor(float(variance)))
        self.eps = eps

    def set_params(self, lengthscale: torch.Tensor, variance: float):
        self.lengthscale = lengthscale.clone().to(self.lengthscale.device)
        self.variance = torch.tensor(float(variance), device=self.lengthscale.device)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        D2 = pairwise_sqdist(X, Y, self.lengthscale + self.eps)
        r = torch.sqrt(D2 + self.eps)
        sqrt5r = math.sqrt(5.0) * r
        term = (1.0 + sqrt5r + 5.0 * D2 / 3.0) * torch.exp(-sqrt5r)
        return self.variance * term


# -----------------------------
# Output reconstruction χ (Eq. (1.7)/(2.13))
# -----------------------------

@dataclass
class OutputKernelParams:
    lengthscale: torch.Tensor     # (d_out,)
    variance: float               # σ_out^2
    rho: float                    # Tikhonov / nugget


class OutputReconstructor(nn.Module):
    """
    χ(y_•)(x) ≈ k_out(x, X_out) (k_out(X_out, X_out) + ρ I)^{-1} y_•

    - k_out: scalar kernel on output coordinates.
    - X_out: (m, d_out) fixed output sensor coordinates.
    """
    def __init__(self,
                 X_out: torch.Tensor,
                 kernel: Literal['rbf', 'rq', 'm52'] = 'm52',
                 init_lengthscale: Optional[torch.Tensor] = None,
                 init_variance: float = 1.0,
                 init_rho: float = 1e-6,
                 eps: float = 1e-12,
                 device: Optional[torch.device] = None):
        super().__init__()
        device = device or X_out.device
        d_out = X_out.shape[1]
        if init_lengthscale is None:
            init_lengthscale = torch.full((d_out,), 0.2, dtype=X_out.dtype, device=device)  # heuristic
        self.register_buffer('X_out', X_out.to(device))
        self.kernel_name = kernel
        if kernel == 'rbf':
            self.kernel = RBFKernel(init_lengthscale, variance=init_variance)
        elif kernel == 'rq':
            self.kernel = RationalQuadraticKernel(init_lengthscale, alpha=1.0, variance=init_variance)
        else:
            self.kernel = Matern52Kernel(init_lengthscale, variance=init_variance)
        self.rho = float(init_rho)
        self.eps = eps
        self._L = None  # Cholesky factor of C = K_out + ρ I

    @property
    def params(self) -> OutputKernelParams:
        if isinstance(self.kernel, RationalQuadraticKernel):
            var = float(self.kernel.variance.item())
        else:
            var = float(self.kernel.variance.item())
        return OutputKernelParams(lengthscale=self.kernel.lengthscale.detach().clone(),
                                  variance=var,
                                  rho=float(self.rho))

    def _make_C(self) -> torch.Tensor:
        K = self.kernel(self.X_out, self.X_out)  # (m, m)
        m = K.shape[0]
        K = (K + (self.rho + self.eps) * torch.eye(m, dtype=K.dtype, device=K.device))
        return K

    def factorize(self):
        """Compute and cache the Cholesky of C := K_out + ρ I."""
        C = self._make_C()
        # Jitter for numerical stability, if needed
        jitter = self.eps
        for _ in range(5):
            try:
                self._L = torch.linalg.cholesky(C + jitter * torch.eye(C.size(0), device=C.device, dtype=C.dtype))
                return
            except RuntimeError:
                jitter *= 10.0
        # Last resort: add larger jitter
        self._L = torch.linalg.cholesky(C + 1e-6 * torch.eye(C.size(0), device=C.device, dtype=C.dtype))

    def log_marginal_likelihood(self, Y: torch.Tensor) -> torch.Tensor:
        """
        GP log-ML for output kernel hyperparams given samples {y_i}_{i=1..N} at X_out:
            L = - N/2 log|C| - 1/2 sum_i y_i^T C^{-1} y_i - N m/2 log(2π)
        where C = k_out(X_out,X_out) + ρ I.
        Y: (N, m)
        """
        Y = _to_2d(Y)  # (N, m)
        N, m = Y.shape
        C = self._make_C()  # (m,m)
        C = 0.5 * (C + C.T)
        # Cholesky
        L = _chol_with_jitter(C, init=max(self.eps, 1e-12), max_tries=8)
        # Solve C^{-1} y_i for all i at once: (m,N)
        Yd = Y.to(C.device)
        alpha = torch.cholesky_solve(Yd.T, L)  # (m,N)
        quad = (Yd.T * alpha).sum()            # sum_i y_i^T C^{-1} y_i
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        const = N * m * math.log(2.0 * math.pi)
        return -0.5 * (N * logdet + quad + const)

    def mle_fit(self,
                Y: torch.Tensor,
                max_iter: int = 80,
                lr: float = 0.2,
                optimize_alpha: bool = True):
        # log-params
        dev   = self.kernel.lengthscale.device
        dtype = self.kernel.lengthscale.dtype
        lengthscale = nn.Parameter(self.kernel.lengthscale.log().detach().clone().to(dev))
        variance    = nn.Parameter(self.kernel.variance.log().detach().clone().to(dev))
        rho         = nn.Parameter(torch.tensor(math.log(max(self.rho, 1e-9)), dtype=dtype, device=dev))
        if isinstance(self.kernel, RationalQuadraticKernel) and optimize_alpha:
            alpha = nn.Parameter(torch.tensor(math.log(max(self.kernel.alpha, 1e-6)), dtype=dtype, device=dev))
        else:
            alpha = None

        params = [lengthscale, variance, rho] + ([alpha] if alpha is not None else [])
        opt = torch.optim.Adam(params, lr=lr)

        best_val = -1e99
        best_tuple = None

        for _ in range(max_iter):
            opt.zero_grad(set_to_none=True)
            ls  = torch.exp(lengthscale)
            var = torch.exp(variance).clamp_min(1e-12)

            if isinstance(self.kernel, RationalQuadraticKernel):
                a_val = float(torch.exp(alpha).clamp_min(1e-6).item()) if alpha is not None else float(self.kernel.alpha)
                self.kernel.set_params(ls, a_val, float(var.item()))
            else:
                self.kernel.set_params(ls, float(var.item()))

            self.rho = float(torch.exp(rho).clamp_min(1e-12).item())

            L = self.log_marginal_likelihood(Y)
            (-L).backward()
            opt.step()

            if L.item() > best_val:
                best_val = L.item()
                best_tuple = (ls.detach().clone(),
                              float(var.item()),
                              float(self.rho),
                              (a_val if isinstance(self.kernel, RationalQuadraticKernel) else None))

        # restore best
        ls_best, var_best, rho_best, alpha_best = best_tuple
        if isinstance(self.kernel, RationalQuadraticKernel):
            self.kernel.set_params(ls_best, float(alpha_best if alpha_best is not None else self.kernel.alpha), float(var_best))
        else:
            self.kernel.set_params(ls_best, float(var_best))
        self.rho = float(rho_best)

        self.factorize()

    def solve_C(self, RHS: torch.Tensor) -> torch.Tensor:
        """
        Solve C x = RHS where C = k_out(X_out,X_out) + ρ I.
        RHS: (..., m)
        Return: (..., m)
        """
        if self._L is None:
            self.factorize()
        L = self._L
        # Move RHS to 2d for solve, then restore shape
        shape = RHS.shape
        m = shape[-1]
        R = RHS.reshape(-1, m).T.to(L.device)  # (m, batch)
        sol = torch.cholesky_solve(R, L)      # (m, batch)
        return sol.T.reshape(shape)

    def reconstruct(self, x_query: torch.Tensor, y_meas: torch.Tensor) -> torch.Tensor:
        """
        Apply χ to an output measurement vector y_meas at the sensors X_out, producing
        predictions on x_query points.
        x_query: (P, d_out) OR (B, P, d_out) for per-batch queries
        y_meas:  (m,) OR (B, m)
        Return:  (P,) or (B, P)
        """
        if y_meas.dim() == 1:
            y_meas = y_meas.unsqueeze(0)
        B, m = y_meas.shape

        coef = self.solve_C(y_meas)               # (B, m)
        if x_query.dim() == 2:
            x_query = x_query.unsqueeze(0).expand(B, -1, -1)  # (B, P, d)

        # batched Kqx @ coef
        # Kqx: (B, P, m), coef: (B, m) -> out: (B, P)
        K_list = []
        for b in range(B):
            K_list.append(self.kernel(x_query[b], self.X_out))  # (P, m)
        Kqx = torch.stack(K_list, dim=0)
        out = (Kqx * coef.unsqueeze(1)).sum(dim=-1)
        return out if out.size(0) > 1 else out[0]

# ─────────────────────────────────────────────────────────
# Input Reconstructor ψ̃ : test-time measurement → training input-sensor space
# ─────────────────────────────────────────────────────────
class InputReconstructor(nn.Module):
    """
    ψ̃: test-time measurements at X_meas -> lift to training input-sensor space X_in

    u^(x) = Q(x, X_meas) [Q(X_meas, X_meas) + ρ I]^{-1} U_meas
    U_train = u^(X_in)

    Args
    ----
    X_in : (N_in, d_in)   training-time input sensor locations (normalized coords)
    kernel : 'rbf' | 'rq' | 'm52'
    rho : small nugget for numerical stability (default 1e-8)
    """
    def __init__(self,
                 X_in: torch.Tensor,
                 kernel: Literal['rbf', 'rq', 'm52'] = 'm52',
                 init_lengthscale: Optional[torch.Tensor] = None,
                 init_variance: float = 1.0,
                 rho: float = 1e-5,
                 device: Optional[torch.device] = None):
        super().__init__()
        device = device or X_in.device
        self.register_buffer('X_in', X_in.to(device))
        d = X_in.shape[1]
        self.rho = float(rho)

        if init_lengthscale is None:
            init_lengthscale = torch.full((d,), 0.2, dtype=X_in.dtype, device=device)

        if kernel == 'rbf':
            self.Q = RBFKernel(init_lengthscale, variance=init_variance)
        elif kernel == 'rq':
            self.Q = RationalQuadraticKernel(init_lengthscale, alpha=1.0, variance=init_variance)
        else:
            self.Q = Matern52Kernel(init_lengthscale, variance=init_variance)

    @torch.no_grad()
    def lift(self, X_meas: torch.Tensor, U_meas: torch.Tensor) -> torch.Tensor:
        """
        X_meas : (B, N_meas, d_in)  normalized input coordinates at test-time
        U_meas : (B, N_meas)       measurement vector on X_meas (e.g., current x_t)
        Return : (B, N_in)         values lifted onto training input sensors X_in
        """
        if X_meas.dim() != 3:
            raise ValueError("X_meas must be (B, N_meas, d_in)")
        if U_meas.dim() != 2:
            raise ValueError("U_meas must be (B, N_meas)")

        B, N_meas, d = X_meas.shape
        N_in = self.X_in.shape[0]
        out = U_meas.new_zeros((B, N_in))

        eye_cache = None
        for b in range(B):
            Xm = X_meas[b]                            # (N_meas, d)
            Um = U_meas[b].view(-1, 1)                # (N_meas, 1)
            C  = self.Q(Xm, Xm)                       # (N_meas, N_meas)
            if eye_cache is None or eye_cache.shape[0] != N_meas:
                eye_cache = torch.eye(N_meas, device=Xm.device, dtype=Xm.dtype)
            C  = C + self.rho * eye_cache
            L  = torch.linalg.cholesky(C)             # Cholesky
            coef = torch.cholesky_solve(Um, L)        # (N_meas, 1)
            K_inm = self.Q(self.X_in, Xm)             # (N_in, N_meas)
            out[b] = (K_inm @ coef).view(-1)          # (N_in,)
        return out
            
# -----------------------------
# Input-side KRR f̄ with exact and Nyström solvers
# -----------------------------

@dataclass
class InputKernelParams:
    lengthscale: torch.Tensor     # (d_in,) (including time if used)
    variance: float               # σ_in^2
    lam: float                    # λ ≥ 0 (ridge / nugget)
    alpha_rq: Optional[float] = None  # for RationalQuadratic

class LinearKernel(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.register_buffer('scale', torch.tensor(float(scale)))
    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        d = X.size(1)
        return (self.scale**2 / max(d,1)) * (X @ Y.T)    

class InputRegressor(nn.Module):
    """
    Implements ŷ_out^meas = S(Ũ, 𝚄̃) α   with α = ( S(𝚄̃,𝚄̃) + λ I )^{-1} 𝚅

    - Ũ ∈ R^{N×d_in} are the input features φ(u) optionally concatenated with time t (product kernel).
    - S: scalar kernel (RBF, RQ, or Matérn 5/2) with anisotropic lengthscales.

    Exact solver: Cholesky per minibatch (faithful to the paper).
    Nyström solver: optional O(N M^2 + M^3) per batch using M anchors Z.
    """
    def __init__(self,
                 kernel: Literal['rbf', 'rq', 'm52'] = 'rq',
                 init_lengthscale: Optional[torch.Tensor] = None,
                 init_variance: float = 1.0,
                 init_lambda: float = 1e-6,
                 eps: float = 1e-12,
                 device: Optional[torch.device] = None,
                 *,
                 use_product_time: bool = True,
                 time_kernel: Literal['rbf', 'rq', 'm52'] = 'rbf',
                 init_time_lengthscale: Optional[torch.Tensor] = None,
                 init_time_variance: float = 1.0):
        super().__init__()
        self.kernel_name = kernel
        self.eps = eps
        self.device = device or torch.device('cpu')

        # 주 커널(입력 특징용)
        self.kernel: nn.Module = None
        self._init_lengthscale = init_lengthscale
        self._init_variance = init_variance
        self._init_lambda = init_lambda

        # 시간 제품커널 옵션
        self.use_product_time = bool(use_product_time)
        self.time_kernel_name = time_kernel
        self._time_kernel: Optional[nn.Module] = None
        self._init_time_lengthscale = init_time_lengthscale
        self._init_time_variance   = init_time_variance

        self.register_buffer('U_mu',  None)   # (1, d_in)
        self.register_buffer('U_std', None)   # (1, d_in)
        self.register_buffer('spatial_shrink', None)

    def _fit_standardizer(self, U: torch.Tensor):
        mu_x  = U[:, :-1].mean(dim=0, keepdim=True)
        std_x = U[:, :-1].std(dim=0, keepdim=True).clamp_min(1e-6)
        mu_t  = U[:, -1:].mean(dim=0, keepdim=True)
        std_t = U[:, -1:].std(dim=0, keepdim=True).clamp_min(1e-6)
        self.U_mu  = torch.cat([mu_x, mu_t], dim=1)
        self.U_std = torch.cat([std_x, std_t], dim=1)
        d_spatial = U.size(1) - 1
        self.spatial_shrink = torch.tensor(float(d_spatial)**0.5, device=U.device)

    def _normU(self, X: torch.Tensor) -> torch.Tensor:
        if (self.U_mu is None) or (self.U_std is None):
            return X
        Z = (X - self.U_mu) / self.U_std
        if self.spatial_shrink is not None and Z.size(1) >= 2:
            Z = Z.clone()
            Z[:, :-1] = Z[:, :-1] / self.spatial_shrink
        return Z               

    def _make_scalar_kernel(self, name: str, lengthscale, variance):
        if name == 'linear':
            return LinearKernel(scale=1.0)
        if name == 'rbf':
            return RBFKernel(lengthscale, variance=variance)
        elif name == 'rq':
            return RationalQuadraticKernel(lengthscale, alpha=1.0, variance=variance)
        else:
            return Matern52Kernel(lengthscale, variance=variance)       

    def _ensure_kernel(self, d_in: int):
        if self.kernel is not None:
            return
        # if product-time: 입력 특징 차원 = d_in-1, 시간 차원 = 1
        if self.use_product_time:
            if d_in < 2:
                raise ValueError("use_product_time=True 이면 입력에 시간 열(t)이 포함되어야 합니다 (d_in>=2).")
            ls_in = self._init_lengthscale or torch.full((d_in - 1,),
                        1.0, dtype=torch.get_default_dtype(), device=self.device)
            self.kernel = self._make_scalar_kernel(self.kernel_name, ls_in, self._init_variance)

            ls_t = self._init_time_lengthscale or torch.ones(1, dtype=torch.get_default_dtype(), device=self.device)
            self._time_kernel = self._make_scalar_kernel(self.time_kernel_name, ls_t, self._init_time_variance)
        else:
            ls = self._init_lengthscale or torch.full((d_in,),
                        1.0, dtype=torch.get_default_dtype(), device=self.device)
            self.kernel = self._make_scalar_kernel(self.kernel_name, ls, self._init_variance)

        self.lam = float(self._init_lambda)

    # 전체 커널: S_total((U,t),(U',t')) = S_in(U,U') * k_t(t,t') (옵션)
    def kernel_total(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = _to_2d(X); Y = _to_2d(Y)
        X = self._normU(X)
        Y = self._normU(Y)
        if not self.use_product_time:
            return self.kernel(X, Y)
        Xin, Xt = X[:, :-1], X[:, -1:].contiguous()
        Yin, Yt = Y[:, :-1], Y[:, -1:].contiguous()
        return self.kernel(Xin, Yin) * self._time_kernel(Xt, Yt)       

    @property
    def params(self) -> InputKernelParams:
        if self.kernel is None:
            raise RuntimeError("Kernel not initialized yet.")
        alpha_rq = self.kernel.alpha if isinstance(self.kernel, RationalQuadraticKernel) else None
        return InputKernelParams(lengthscale=self.kernel.lengthscale.detach().clone(),
                                 variance=float(self.kernel.variance.item()),
                                 lam=float(self.lam),
                                 alpha_rq=alpha_rq)

    # ---------- Exact solver (Cholesky) ----------
    def exact_alpha(self, U: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve α = (K + λ I)^{-1} V  with  K = S(U,U).

        U: (N, d_in), V: (N, m)
        Return: α (N, m), K (N, N), L (Cholesky of K+λI)
        """
        U = _to_2d(U); V = _to_2d(V)
        N = U.shape[0]
        self._ensure_kernel(U.shape[1])
        K = self.kernel_total(U, U)  # (N,N)
        A = K + (self.lam + self.eps) * torch.eye(N, device=K.device, dtype=K.dtype)
        jitter = self.eps
        for _ in range(5):
            try:
                L = torch.linalg.cholesky(A + jitter * torch.eye(N, device=A.device, dtype=A.dtype))
                break
            except RuntimeError:
                jitter *= 10.0
        else:
            L = torch.linalg.cholesky(A + 1e-6 * torch.eye(N, device=A.device, dtype=A.dtype))
        alpha = torch.cholesky_solve(V.to(L.device), L)  # (N,m)
        return alpha, K, L

    def predict_meas_from_alpha(self, K: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Predict ŷ_out^meas at U (training points):  ŷ = K α   (with K=S(U,U), not including λI)
        """
        return K @ alpha

    def predict_meas(self, U_query: torch.Tensor, U_dict: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Predict ŷ_out^meas at new inputs U_query given dictionary U_dict and α.

        U_query: (Q, d_in), U_dict: (N, d_in), alpha: (N, m)
        Return: (Q, m)
        """
        Kq = self.kernel_total(U_query, U_dict)  # (Q,N)
        return Kq @ alpha

    # ---------- Nyström low-rank solver ----------
    def nystrom_alpha(self,
                      U: torch.Tensor,
                      V: torch.Tensor,
                      Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nyström approximation with anchors Z (M×d):
          K ≈ W C^{-1} W^T  where W=S(U,Z), C=S(Z,Z).
          (λI + K)^{-1} ≈ λ^{-1}I - λ^{-1} W ( C + λ^{-1} W^T W )^{-1} W^T λ^{-1}.

        Returns α ≈ (λI + K)^{-1} V and (optionally) helper matrix for reuse.
        """
        U = _to_2d(U); V = _to_2d(V); Z = _to_2d(Z)
        self._ensure_kernel(U.shape[1])
        N, m = V.shape
        M = Z.shape[0]
        lam = float(self.lam + self.eps)

        W  = self.kernel_total(U, Z)   # (N,M)
        C  = self.kernel_total(Z, Z)   # (M,M)
        WT_W = W.T @ W
        B = C + (1.0 / lam) * WT_W

        jitter = self.eps
        for _ in range(5):
            try:
                LB = torch.linalg.cholesky(B + jitter * torch.eye(M, device=B.device, dtype=B.dtype))
                break
            except RuntimeError:
                jitter *= 10.0
        else:
            LB = torch.linalg.cholesky(B + 1e-6 * torch.eye(M, device=B.device, dtype=B.dtype))

        V = V.to(W.device)
        Vinv = (1.0 / lam) * V
        rhs  = W.T @ Vinv
        tmp  = torch.cholesky_solve(rhs, LB)
        corr = W @ tmp
        alpha = Vinv - (1.0 / lam) * corr
        return alpha, LB

    # ---------- MLE for input kernel ----------
    def log_marginal_likelihood(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        GP log-ML for Γ(U,U') = S_total(U,U') I_m (product-in-time if enabled):
            L = - (m/2) log|K+λI| - 1/2 Tr( V^T (K+λI)^{-1} V ) - (mN/2) log(2π)
        """
        U = _to_2d(U); V = _to_2d(V)
        N, m = V.shape
        self._ensure_kernel(U.shape[1])
        # BUGFIX: use full product kernel (space × time) when enabled
        K = self.kernel_total(U, U)     
        A = 0.5 * (K + K.T)                            # (N, N)
        A = A + (self.lam + self.eps) * torch.eye(N, device=A.device, dtype=A.dtype)
        L = _chol_with_jitter(A, init=max(self.eps, 1e-12), max_tries=8)
        Vd = V.to(L.device)
        alpha = torch.cholesky_solve(Vd, L)                          # (N, m)
        quad = (Vd * alpha).sum()
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        const = N * m * math.log(2.0 * math.pi)
        return -0.5 * (m * logdet + quad + const)

    def mle_fit(self,
                U: torch.Tensor,
                V: torch.Tensor,
                max_iter: int = 120,
                lr: float = 0.2,
                optimize_alpha: bool = True,
                optimize_time: bool = True):
        """
        Maximize GP marginal likelihood to fit (ℓ_in, σ_in, λ) and, if enabled,
        time-kernel hyperparams (ℓ_t, σ_t[, α_t]).
        """
        U = _to_2d(U); V = _to_2d(V)
        d_in = U.shape[1]
        self._ensure_kernel(d_in)

        # space kernel params (log-parametrized)
        dev = self.kernel.lengthscale.device
        dtype = self.kernel.lengthscale.dtype
        lengthscale = nn.Parameter(self.kernel.lengthscale.log().detach().clone().to(dev))
        variance    = nn.Parameter(self.kernel.variance.log().detach().clone().to(dev))
        lam         = nn.Parameter(torch.tensor(math.log(max(self.lam, 1e-9)), dtype=dtype, device=dev))
        alpha = None
        if isinstance(self.kernel, RationalQuadraticKernel) and optimize_alpha:
            alpha = nn.Parameter(torch.tensor(math.log(max(self.kernel.alpha, 1e-6)), dtype=dtype, device=dev))

        params = [lengthscale, variance, lam] + ([alpha] if alpha is not None else [])

        # time kernel params if product-in-time is on
        use_time = self.use_product_time and (self._time_kernel is not None) and optimize_time
        if use_time:
            ls_t  = nn.Parameter(self._time_kernel.lengthscale.log().detach().clone().to(dev))
            var_t = nn.Parameter(self._time_kernel.variance.log().detach().clone().to(dev))
            params += [ls_t, var_t]
            a_t = None
            if isinstance(self._time_kernel, RationalQuadraticKernel):
                a_t = nn.Parameter(torch.tensor(math.log(max(self._time_kernel.alpha, 1e-6)),
                                                dtype=dtype, device=dev))
                params.append(a_t)

        opt = torch.optim.Adam(params, lr=lr)

        best_val = -1e99
        best_pack = None

        for _ in range(max_iter):
            opt.zero_grad(set_to_none=True)

            # unpack log-params
            ls_sp  = torch.exp(lengthscale)
            var_sp = torch.exp(variance).clamp_min(1e-12)
            self.lam = float(torch.exp(lam).clamp_min(1e-12).item())

            # set module params (space)
            if isinstance(self.kernel, RationalQuadraticKernel):
                a_sp = float(torch.exp(alpha).clamp_min(1e-6).item()) if alpha is not None else self.kernel.alpha
                self.kernel.set_params(ls_sp, a_sp, float(var_sp.item()))
            else:
                self.kernel.set_params(ls_sp, float(var_sp.item()))

            # set module params (time)
            cur_ls_t = cur_var_t = cur_a_t = None
            if use_time:
                cur_ls_t  = torch.exp(ls_t)
                cur_var_t = torch.exp(var_t).clamp_min(1e-12)
                if isinstance(self._time_kernel, RationalQuadraticKernel) and a_t is not None:
                    cur_a_t = float(torch.exp(a_t).clamp_min(1e-6).item())
                    self._time_kernel.set_params(cur_ls_t, cur_a_t, float(cur_var_t.item()))
                else:
                    self._time_kernel.set_params(cur_ls_t, float(cur_var_t.item()))

            # objective
            L = self.log_marginal_likelihood(U, V)
            (-L).backward()
            opt.step()

            if L.item() > best_val:
                best_val = L.item()
                best_pack = {
                    "ls_sp": ls_sp.detach().clone(),
                    "var_sp": float(var_sp.item()),
                    "lam": float(self.lam),
                }
                if isinstance(self.kernel, RationalQuadraticKernel):
                    best_pack["a_sp"] = float(a_sp)
                if use_time:
                    best_pack["ls_t"]  = cur_ls_t.detach().clone()
                    best_pack["var_t"] = float(cur_var_t.item())
                    if cur_a_t is not None:
                        best_pack["a_t"] = float(cur_a_t)

        # ---- restore best ----
        if isinstance(self.kernel, RationalQuadraticKernel):
            alpha_best = best_pack.get("a_sp", getattr(self.kernel, "alpha", 1.0))
            self.kernel.set_params(best_pack["ls_sp"], float(alpha_best), float(best_pack["var_sp"]))
        else:
            self.kernel.set_params(best_pack["ls_sp"], float(best_pack["var_sp"]))
        self.lam = float(best_pack["lam"])

        if use_time:
            if isinstance(self._time_kernel, RationalQuadraticKernel):
                alpha_t_best = best_pack.get("a_t", getattr(self._time_kernel, "alpha", 1.0))
                self._time_kernel.set_params(best_pack["ls_t"], float(alpha_t_best), float(best_pack["var_t"]))
            else:
                self._time_kernel.set_params(best_pack["ls_t"], float(best_pack["var_t"]))


# -----------------------------
# Full HDM Backbone:  Ḡ = χ ∘ f̄ ∘ φ  (Eq. (2.5))
# -----------------------------

class KernelOpHDM(nn.Module):
    """
    Drop-in HDM backbone implementing the exact paper formulas with per-minibatch recomputation.

    Constructor args
    ---------------
    X_in:  (n, d_in_space)    fixed input sensor locations used by φ (if you provide φ(U) vectors yourself,
                              you can ignore X_in and pass U directly to forward).
    X_out: (m, d_out)         fixed output sensor locations used by varphi and χ.
    in_kernel/out_kernel:     one of {'rq', 'rbf', 'm52'}
    solver:                   'exact' (Cholesky) or 'nystrom'
    nystrom_anchors:          Optional (M, d_in) anchor set for Nyström; if None and solver='nystrom',
                              will select M random rows from current minibatch each call.
    use_time:                 If True, append scalar t as last column to U and use anisotropic lengthscale
                              (i.e., product kernel S_in(U,U') * k_t(t,t')).
    """
    def __init__(self,
                 X_in: Optional[torch.Tensor],
                 X_out: torch.Tensor,
                 in_kernel: Literal['rq','rbf','m52']='rq',
                 out_kernel: Literal['rq','rbf','m52']='m52',
                 solver: Literal['exact','nystrom']='exact',
                 nystrom_anchors: Optional[torch.Tensor]=None,
                 use_time: bool=True,
                 device: Optional[torch.device]=None,
                 *,
                 use_product_time: bool=True,
                 time_kernel: Literal['rbf','rq','m52']='rbf',
                 # --- NEW: 초기 하이퍼(옵션) ---
                 in_init_lengthscale: Optional[torch.Tensor]=None,
                 in_init_variance: float = 1.0,
                 in_init_lambda: float = 1e-6,
                 time_init_lengthscale: Optional[torch.Tensor]=None,
                 time_init_variance: float = 1.0,
                 out_init_lengthscale: Optional[torch.Tensor]=None,
                 out_init_variance: float = 1.0,
                 out_init_rho: float = 1e-6,
                 # --- NEW: ψ̃(inlift) 옵션 ---
                 inlift_enable: bool = True,
                 inlift_kernel: Literal['rbf','rq','m52'] = 'm52',
                 inlift_init_lengthscale: Optional[torch.Tensor] = None,
                 inlift_init_variance: float = 1.0,
                 inlift_rho: float = 1e-8):
        super().__init__()
        device = X_out.device if device is None else device

        self.register_buffer('X_in', X_in if X_in is not None else None)
        self.solver = solver
        self.register_buffer('Z_nys', None if nystrom_anchors is None else nystrom_anchors.to(device))
        self.use_time = bool(use_time)
        self.device = device

        # ---- χ (출력 보간) : 초기 하이퍼 반영 ----
        d_out = X_out.shape[1]
        if out_init_lengthscale is None:
            out_init_lengthscale = torch.full((d_out,), 0.2, dtype=X_out.dtype, device=device)
        elif not torch.is_tensor(out_init_lengthscale):
            out_init_lengthscale = torch.as_tensor(out_init_lengthscale, dtype=X_out.dtype, device=device).view(d_out)
        self.out = OutputReconstructor(
            X_out=X_out.to(device),
            kernel=out_kernel,
            init_lengthscale=out_init_lengthscale,
            init_variance=float(out_init_variance),
            init_rho=float(out_init_rho),
            device=device
        )

        # ---- 입력측 KRR : product-in-time + 초기 하이퍼 반영 ----
        def _as_tensor_maybe(v):  # 스칼라/리스트 허용
            if v is None or torch.is_tensor(v):
                return v
            return torch.as_tensor(v, dtype=X_out.dtype, device=device)

        self.inreg = InputRegressor(
            kernel=in_kernel,
            init_lengthscale=_as_tensor_maybe(in_init_lengthscale),
            init_variance=float(in_init_variance),
            init_lambda=float(in_init_lambda),
            device=device,
            use_product_time=use_product_time,
            time_kernel=time_kernel,
            init_time_lengthscale=_as_tensor_maybe(time_init_lengthscale),
            init_time_variance=float(time_init_variance)
        )

        # ---- ψ̃(inlift) : 메시/측정 불변 입력 복원 옵션 ----
        self.inlift = None
        if inlift_enable and (X_in is not None):
            d_in = X_in.shape[1]
            if inlift_init_lengthscale is None:
                inlift_init_lengthscale = torch.full((d_in,), 0.2, dtype=X_in.dtype, device=device)
            elif not torch.is_tensor(inlift_init_lengthscale):
                inlift_init_lengthscale = torch.as_tensor(inlift_init_lengthscale, dtype=X_in.dtype, device=device).view(d_in)
            self.inlift = InputReconstructor(
                X_in=X_in.to(device),
                kernel=inlift_kernel,
                init_lengthscale=inlift_init_lengthscale,
                init_variance=float(inlift_init_variance),
                rho=float(inlift_rho),
                device=device
            )

        # 전역 사전 캐시
        self.register_buffer('_U_dict', None)
        self.register_buffer('_alpha_dict', None)

    @torch.no_grad()
    def register_dictionary(self, U_all: torch.Tensor, V_all: torch.Tensor,
                            *, anchors: Optional[torch.Tensor] = None):
        """
        전역 사전(학습 사전 전체)을 등록하고 alpha를 캐시한다.
        U_all: (N, d_in[+1])  V_all: (N, m)
        """
        U_all = _to_2d(U_all).to(self.device)
        V_all = _to_2d(V_all).to(self.device)

        self.inreg._fit_standardizer(U_all)

        if self.use_time is False and U_all.shape[1] == 1:
            pass  # nothing

        if self.solver == 'exact':
            alpha, _, _ = self.inreg.exact_alpha(U_all, V_all)
        else:
            Z = anchors if anchors is not None else (self.Z_nys if self.Z_nys is not None else U_all[: min(64, U_all.size(0))])
            alpha, _ = self.inreg.nystrom_alpha(U_all, V_all, Z)

        self._U_dict = U_all
        self._alpha_dict = alpha

        with torch.no_grad():
            K = self.inreg.kernel_total(self._U_dict, self._U_dict).detach()
            diag = torch.diagonal(K).mean().item()
            off  = (K - torch.diag(torch.diagonal(K))).mean().item()
            stdK = K.std().item()
            print(f"[KERNEL DEBUG] mean(diag)={diag:.4f}, mean(off)={off:.4f}, std(K)={stdK:.4f}, ratio(off/diag)={off/max(diag,1e-12):.3f}")        

    @torch.no_grad()
    def query(self, U_query: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
        """
        전역 사전이 등록되어 있을 때 빠른 추론:
          y^meas_hat(U_query) = S(U_query, U_dict) alpha_dict
          s_hat = χ(y^meas_hat)(x_query)
        """
        if (self._U_dict is None) or (self._alpha_dict is None):
            raise RuntimeError("Dictionary is not registered. Call register_dictionary first.")
        U_query = _to_2d(U_query).to(self.device)
        y_meas_hat = self.inreg.predict_meas(U_query, self._U_dict, self._alpha_dict)  # (Q,m)
        return self.out.reconstruct(x_query, y_meas_hat)                

    # ---------- Hyperparam fits (offline / pre-pass) ----------
    def fit_output_kernel_hyperparams(self, Y: torch.Tensor,
                                      max_iter: int = 100, lr: float = 0.2, optimize_alpha: bool = False):
        """
        Fit (ℓ_out, σ_out, ρ) by maximizing the GP marginal likelihood using stacked outputs Y (N×m).
        After fitting, the (K_out + ρI) factorization is cached.
        """
        self.out.mle_fit(Y, max_iter=max_iter, lr=lr, optimize_alpha=optimize_alpha)

    def fit_input_kernel_hyperparams(self, U: torch.Tensor, V: torch.Tensor,
                                     max_iter: int = 150, lr: float = 0.2, optimize_alpha: bool = False, optimize_time: bool = True,):
        """
        Fit (ℓ_in, σ_in, λ) by maximizing the GP marginal likelihood using dictionary (U,V).
        """
        if self.use_time and U.dim() == 1:
            U = U[:, None]
        self.inreg.mle_fit(U, V, max_iter=max_iter, lr=lr, optimize_alpha=optimize_alpha, optimize_time=optimize_time,)

    # ---------- Main forward (per-minibatch/time) ----------
    def forward(self,
                U_batch: torch.Tensor,
                V_batch: Optional[torch.Tensor],
                x_query: torch.Tensor,
                t_batch: Optional[torch.Tensor] = None,
                anchors: Optional[torch.Tensor] = None,
                x_in_meas: Optional[torch.Tensor] = None):
        """
        Per-minibatch forward pass following Eqs. (2.12) and (2.13).

        Args
        ----
        U_batch : (B, N_meas) or (B, d_in)    test-time input measurements (e.g., current x_t on measured coords)
        V_batch : (B, m) or None              output measurements at X_out (e.g., -e); None when using dictionary
        x_query : (B, P, d_out)               query coords for output reconstruction (normalized)
        t_batch : (B,) or (B,1) or None       time; required if use_time=True
        anchors : (M, d_in) or None           optional Nyström anchors
        x_in_meas : (B, N_meas, d_in) or None normalized input coords used to measure U_batch
        """
        # Fast path: dictionary-based inference
        if (self._U_dict is not None) and (self._alpha_dict is not None) and (V_batch is None):
            if self.use_time and t_batch is None:
                raise ValueError("t_batch is required when use_time=True")

            Uq = U_batch
            # Lift to training input sensor space if ψ̃ is available and measurement coords are provided
            if (self.inlift is not None) and (x_in_meas is not None):
                Uq = self.inlift.lift(x_in_meas, U_batch)          # (B, N_in)

            if self.use_time:
                Uq = torch.cat([Uq, t_batch.reshape(-1, 1).to(Uq)], dim=1)  # (B, N_in+1)

            return self.query(Uq, x_query)

        # Otherwise, compute per-batch α and predict y^meas
        if self.use_time and t_batch is None:
            raise ValueError("t_batch is required when use_time=True")

        Uin = U_batch
        if (self.inlift is not None) and (x_in_meas is not None):
            Uin = self.inlift.lift(x_in_meas, U_batch)             # (B, N_in)

        if self.use_time:
            Uin = torch.cat([Uin, t_batch.reshape(-1, 1).to(Uin)], dim=1)  # (B, N_in+1)

        if self.solver == 'exact':
            alpha, K, _ = self.inreg.exact_alpha(Uin, _to_2d(V_batch))
            y_meas_hat = self.inreg.predict_meas_from_alpha(K, alpha)       # (B, m)
        else:
            Z = anchors if anchors is not None else (self.Z_nys if self.Z_nys is not None else Uin[: min(64, Uin.size(0))])
            alpha, _ = self.inreg.nystrom_alpha(Uin, _to_2d(V_batch), Z)
            W = self.inreg.kernel_total(Uin, Z)
            C = self.inreg.kernel_total(Z, Z)
            LC = torch.linalg.cholesky(C + 1e-12 * torch.eye(C.size(0), device=C.device, dtype=C.dtype))
            z = torch.cholesky_solve((W.T @ alpha), LC)
            y_meas_hat = W @ z

        if x_query.dim() == 3 and x_query.size(1) == self.out.X_out.size(0):
            same = torch.allclose(
                x_query[0, :, :], self.out.X_out.unsqueeze(1),
                atol=1e-6, rtol=1e-6
            )
            print("[KERNEL DEBUG] bypass chi:", bool(same))
            if same:
                return y_meas_hat            

        # χ for continuous reconstruction on x_query
        s_hat = self.out.reconstruct(x_query, y_meas_hat)
        return s_hat

    # Convenience helpers
    @property
    def input_params(self) -> InputKernelParams:
        return self.inreg.params

    @property
    def output_params(self) -> OutputKernelParams:
        return self.out.params

# -----------------------------
# Optional helpers for sensor measurements φ / varphi via bilinear sampling (2D)
# -----------------------------

def sample_field_at_points_2d(field: torch.Tensor,
                              grid_xy: torch.Tensor,
                              points_xy: torch.Tensor) -> torch.Tensor:
    """
    Linearly sample a 2D scalar field on an arbitrary set of points using torch.grid_sample.

    field:      (B, 1, H, W)       values on a Cartesian grid
    grid_xy:    (H, W, 2)          absolute coordinates in [0,1]^2 for each pixel center
    points_xy:  (B, M, 2)          absolute coordinates in [0,1]^2

    Returns     (B, M)             sampled values

    Note: rescaling to [-1,1] is handled internally.
    """
    B, _, H, W = field.shape
    # Build a normalized coordinate grid for grid_sample
    # Convert absolute [0,1] coords → normalized [-1,1] coords
    norm = lambda xy: xy * 2.0 - 1.0
    # Create a dense lookup of normalized coordinates per batch
    # grid_sample expects (B, H_out, W_out, 2). For scattered points, we emulate a 1×M grid.
    points_norm = norm(points_xy).view(B, 1, -1, 2)  # (B,1,M,2)
    # grid_sample requires the source to be in normalized coords; we assume grid_xy is uniform [0,1]^2
    # If your input grid is irregular, pre-warp it with a mapping.
    sampled = F.grid_sample(field, points_norm, align_corners=True, mode='bilinear', padding_mode='border')  # (B,1,1,M)
    return sampled.view(B, -1)