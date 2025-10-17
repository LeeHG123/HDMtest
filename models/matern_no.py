# models/matern_no.py
# -----------------------------------------------------------------------------
# Matern-NO (1D): NUFNO의 스펙트럴 블록만 Matérn-RKHS(Laplacian eigenbasis)로 치환
# -----------------------------------------------------------------------------
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import (
    get_timestep_embedding,
    Lifting,
    Projection,
    default_init,
)
from .mlp import MLP, skip_connection


# ----------------------------- utilities -------------------------------------
def _trapz_weights(x: torch.Tensor) -> torch.Tensor:
    """
    사다리꼴 가중치 (배치별, 비등간격 대응).
    x : (B, N) — 좌표(정규화 전/후 어떤 것이든 단조이면 안전). 정렬 가정 없이 |Δx| 사용.
    return: (B, N) — 구간 가중치
    """
    assert x.dim() == 2, "x must be (B, N)"
    dx = (x[:, 1:] - x[:, :-1]).abs()
    w = torch.zeros_like(x)
    if x.size(1) > 2:
        w[:, 1:-1] = 0.5 * (dx[:, 1:] + dx[:, :-1])
    w[:, 0] = 0.5 * dx[:, 0]
    w[:, -1] = 0.5 * dx[:, -1]
    return w


def _matern_alpha_from_nu(nu: float, d: int = 1) -> float:
    """
    Matérn SPDE:  (λ^2 - Δ)^α u = W,   α = ν + d/2
    """
    return float(nu + 0.5 * d)


# --------------------- Laplacian eigen-basis on [-1, 1] ----------------------
class _Laplace1DBasis(nn.Module):
    """
    [-1,1] 구간의 1D 라플라시안 고유기저.
    - Dirichlet: ψ_k(x) = sin( (k π/2) (x + 1) ),   k=1,2,...,M
      μ_k = (k π/2)^2
    - Neumann  : ψ_k(x) = cos( (k π/2) (x + 1) ),   k=0,1,...,M-1
      μ_k = (k π/2)^2

    입력은 정규화 좌표 x_norm ∈ [-1, 1].
    """
    def __init__(self, M: int, boundary: str = "neumann"):
        super().__init__()
        boundary = str(boundary).lower()
        if boundary not in ("neumann", "dirichlet"):
            raise ValueError("boundary must be 'neumann' or 'dirichlet'.")

        self.boundary = boundary
        self.M = int(M)

        if self.boundary == "dirichlet":
            k = torch.arange(1, self.M + 1, dtype=torch.float32)   # 1..M
        else:
            k = torch.arange(0, self.M, dtype=torch.float32)        # 0..M-1

        # ω_k = (k π/2),  μ_k = ω_k^2
        omega = 0.5 * math.pi * k
        mu = omega ** 2
        self.register_buffer("k_idx", k, persistent=False)
        self.register_buffer("omega", omega, persistent=False)
        self.register_buffer("mu", mu, persistent=False)

    @torch.no_grad()
    def eigenvalues(self) -> torch.Tensor:
        """μ_k = (k π/2)^2, shape (M,)"""
        return self.mu

    def eval(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        ψ(x) 행렬을 계산.
        x_norm: (B, N),  in [-1, 1]
        return: (B, N, M)  with ψ_k(x_n)
        """
        assert x_norm.dim() == 2
        B, N = x_norm.shape
        # t = (π/2) * (x+1)   →  ω_k * (x+1) = (k π/2) (x+1)
        t = 0.5 * math.pi * (x_norm + 1.0)                         # (B, N)
        t = t.unsqueeze(-1) * self.k_idx.view(1, 1, self.M)        # (B, N, M)
        if self.boundary == "dirichlet":
            return torch.sin(t)
        else:
            return torch.cos(t)


# ---------------------- Matérn spectral conv in ψ-basis ----------------------
class MaternSpectralConv1D(nn.Module):
    """
    스펙트럴 블록(1D):
      - 입력/출력: x_feat ∈ R^{B × C_in × N}
      - 기저: Laplace-1D eigenbasis ψ_k(x) (Dirichlet/Neumann)
      - 투영:   c_k = ⟨x_c, ψ_k⟩_L2  (사다리꼴 가중으로 근사; 분해 시 해상도-불변)
      - 곱셈:   ĉ_k = R_k * c_k   (채널-분리/비분리 선택)
      - 복원:   y(x) = Σ_k ĉ_k ψ_k(x)

    Matérn 초기화:
      R_k^{(0)} = (λ^2 + μ_k)^{-α},   α = ν + d/2  (d=1)

    파라미터:
      - separable=True  → 채널별 모드 스칼라 (C × M)
      - separable=False → (C_out × C_in × M)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        *,
        temb_dim: int = 256,
        boundary: str = "neumann",
        matern_nu: float = 1.5,
        matern_ell: float = 1.0,     # ℓ,  λ = 1/ℓ
        separable: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.separable = bool(separable)

        if self.separable and (self.in_channels != self.out_channels):
            raise ValueError("separable=True이면 in_channels == out_channels 이어야 합니다.")

        # 기저 & 고유값
        self.basis = _Laplace1DBasis(M=n_modes, boundary=boundary)

        # Matérn 초기화 계수 R0(μ_k)
        with torch.no_grad():
            mu = self.basis.eigenvalues()                                  # (M,)
            lam = 1.0 / max(1e-12, float(matern_ell))                      # λ = 1/ℓ
            alpha = _matern_alpha_from_nu(float(matern_nu), d=1)           # α = ν + 1/2
            R0 = torch.pow(lam * lam + mu, -alpha)                         # (M,)
        self.register_buffer("R0", R0, persistent=False)

        # 학습 파라미터:  R = R0 * (1 + δ)
        if self.separable:
            self.delta = nn.Parameter(torch.zeros(self.in_channels, n_modes))
        else:
            self.delta = nn.Parameter(torch.zeros(self.out_channels, self.in_channels, n_modes))

        # time-embedding → 채널 shift (NUFNO와 동일한 역할)
        self.tact = nn.SiLU()
        self.tproj = nn.Linear(temb_dim, in_channels)
        self.tproj.weight.data = default_init()(self.tproj.weight.data.shape)
        nn.init.zeros_(self.tproj.bias)

        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1)) if bias else None

    def _spectral_weight(self) -> torch.Tensor:
        """
        현재 모드 승수 R (양/음 제약 없음, R0*(1+δ))
        separable  : (C, M)
        nonsep     : (C_out, C_in, M)
        """
        if self.separable:
            return self.R0.view(1, -1) * (1.0 + self.delta)
        else:
            return self.R0.view(1, 1, -1) * (1.0 + self.delta)

    def forward(self, x_feat: torch.Tensor, z: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_feat : (B, C_in, N)
        z     : (B, N) — NUFNO와 동일 입력(π x_norm). 본 블록에선 x_norm=z/π 사용.
        temb  : (B, C_temb) — 시간 임베딩(채널 시프트에 사용)
        return: (B, C_out, N)
        """
        B, C_in, N = x_feat.shape
        device = x_feat.device
        dtype  = x_feat.dtype

        # 시간조건: 채널별 additive shift
        if temb is not None:
            shift = self.tproj(self.tact(temb))           # (B, C_in)
            x_feat = x_feat + shift.unsqueeze(-1)

        # 좌표/가중치/기저값
        x_norm = (z / math.pi).to(device=device, dtype=dtype)   # (B, N) in [-1,1]
        psi = self.basis.eval(x_norm)                           # (B, N, M)

        # 해상도-불변 투영: 사다리꼴 가중 정규화
        w = _trapz_weights(x_norm)                              # (B, N)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)            # (B, N)
        xw = x_feat * w.unsqueeze(1)                            # (B, C_in, N)

        # c_k = ∫ x ψ_k dx  ≈ Σ_j w_j x_j ψ_k(x_j)
        coeff = torch.bmm(xw, psi)                              # (B, C_in, M)

        # 스펙트럼 곱셈 R_k
        R = self._spectral_weight()                             # (C,M) or (Cout,Cin,M)
        if self.separable:
            # (B, C, M) * (C, M) → (B, C, M)
            ycoeff = coeff * R.unsqueeze(0)
        else:
            # (B, 1, Cin, M),  (1, Cout, Cin, M) → sum_Cin → (B, Cout, M)
            coeff_e = coeff.unsqueeze(1)                        # (B, 1, Cin, M)
            R_e     = R.unsqueeze(0)                            # (1, Cout, Cin, M)
            ycoeff  = (coeff_e * R_e).sum(dim=2)                # (B, Cout, M)

        # 복원: y(x) = Σ_k ĉ_k ψ_k(x)
        psiT = psi.transpose(1, 2)                               # (B, M, N)
        y = torch.bmm(ycoeff, psiT)                              # (B, C_out, N)  (separable면 C_out=C_in)

        if self.bias is not None:
            y = y + self.bias
        return y


# ------------------------------ Matern-NO ------------------------------------
class MaternNO(nn.Module):
    """
    NUFNO 아키텍처를 그대로 따르되,
    spectral_blocks만 MaternSpectralConv1D로 교체.
    나머지 lifting / time-embedding / norm / skip / MLP / projection / forward는 동일.
    """
    def __init__(self, config):
        super().__init__()
        cfg = config.model

        # 필수 하이퍼
        self.n_dim = 1
        self.n_modes = cfg.n_modes
        assert isinstance(self.n_modes, (list, tuple)) and len(self.n_modes) == 1, \
            "MaternNO는 1D만 지원하므로 n_modes는 [int,] 형식이어야 합니다."
        self.hidden_channels = cfg.hidden_channels
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.lifting_channels = cfg.lifting_channels
        self.projection_channels = cfg.projection_channels
        self.n_layers = cfg.n_layers
        self.norm_type = getattr(cfg, "norm", "group_norm")
        self.preactivation = getattr(cfg, "preactivation", True)
        self.skip_type = getattr(cfg, "skip", "soft-gating")
        self.separable = getattr(cfg, "separable", True)

        # --- Matern 하이퍼(추가) ---
        self.boundary = str(getattr(cfg, "matern_boundary", "neumann")).lower()
        self.matern_nu = float(getattr(cfg, "matern_nu", 1.5))
        self.matern_ell = float(getattr(cfg, "matern_ell", 1.0))

        # 시간 임베딩 두 단계 (NUFNO와 동일)
        self.Dense = nn.ModuleList([
            nn.Linear(self.lifting_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        ])
        for layer in self.Dense:
            layer.weight.data = default_init()(layer.weight.data.shape)
            nn.init.zeros_(layer.bias)

        # Lifting / Projection
        self.lifting = Lifting(in_channels=self.in_channels, out_channels=self.hidden_channels, n_dim=1)
        self.projection = Projection(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_dim=1,
            non_linearity=F.gelu,
        )

        # --- 핵심: Matérn 스펙트럴 블록들 ---
        self.spectral_blocks = nn.ModuleList([
            MaternSpectralConv1D(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_modes=self.n_modes[0],
                temb_dim=self.lifting_channels,
                boundary=self.boundary,
                matern_nu=self.matern_nu,
                matern_ell=self.matern_ell,
                separable=self.separable,
                bias=True,
            ) for _ in range(self.n_layers)
        ])

        # Norm / Skip / (선택) MLP
        if self.norm_type is None:
            self.norms = None
        elif self.norm_type == "group_norm":
            self.norms = nn.ModuleList([nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels) for _ in range(self.n_layers)])
        elif self.norm_type == "instance_norm":
            self.norms = nn.ModuleList([nn.InstanceNorm1d(num_features=self.hidden_channels) for _ in range(self.n_layers)])
        else:
            raise ValueError(f"Unsupported norm: {self.norm_type}")

        self.skips = nn.ModuleList([
            skip_connection(self.hidden_channels, self.hidden_channels, n_dim=1, type=self.skip_type)
            for _ in range(self.n_layers)
        ])

        self.use_mlp = True
        self.mlps = nn.ModuleList([
            MLP(in_channels=self.hidden_channels, hidden_channels=int(round(self.hidden_channels * 4.0)),
                dropout=0.0, n_dim=1, temb_dim=self.hidden_channels)
            for _ in range(self.n_layers)
        ]) if self.use_mlp else None

        self.act = nn.SiLU()

    # NUDFT 정규화 항을 쓰는 코드가 NUFNO 전용 κ에 의존하므로,
    # 안전하게 빈 리스트를 반환하도록 API만 맞춰 둡니다.
    def all_kappas(self) -> List[torch.Tensor]:
        return []

    def all_baselines(self) -> List[torch.Tensor]:
        return []

    def all_band_gates(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in=2, N)  — [signal, coord_norm]
        t: (B,)
        return: (B, N)
        """
        B, Cin, N = x.shape

        # 좌표 채널 (러너는 [-1,1] 정규화 값을 넣어줌)
        x_coord_norm = x[:, -1, :]  # (B, N)

        # Lifting
        h = self.lifting(x)         # (B, C_hid, N)

        # 시간 임베딩
        temb = get_timestep_embedding(t, self.lifting_channels)  # (B, C_lift)
        temb = self.Dense[0](temb)
        temb = self.Dense[1](F.silu(temb))
        h = h + temb.unsqueeze(-1)  # (B, C_hid, N)

        # 스펙트럴 블록: z = π * x_norm  (NUFNO와 동일 인터페이스)
        z = math.pi * x_coord_norm  # (B, N)

        for i in range(self.n_layers):
            if self.preactivation:
                h = F.silu(h)
                if self.norms is not None:
                    h = self.norms[i](h)

            h_f = self.spectral_blocks[i](h, z, temb)  # (B, C_hid, N)

            if not self.preactivation and self.norms is not None:
                h_f = self.norms[i](h_f)

            h = h_f + self.skips[i](h)

            if self.use_mlp:
                h = self.mlps[i](h, temb)

        y = self.projection(h).squeeze(1)  # (B, N)
        return y
