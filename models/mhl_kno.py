# models/mhl_kno.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quasirandom import SobolEngine

from .fno import (
    get_timestep_embedding,
    Lifting,
    Projection,
    default_init,
)
from .mlp import MLP, skip_connection

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _trapz_weights_1d(x: torch.Tensor) -> torch.Tensor:
    B, N = x.shape
    if N == 1:
        return torch.ones_like(x)
    dx = (x[:, 1:] - x[:, :-1]).abs()
    w = torch.zeros_like(x)
    w[:, 0]  = 0.5 * dx[:, 0]
    w[:, -1] = 0.5 * dx[:, -1]
    if N > 2:
        w[:, 1:-1] = 0.5 * (dx[:, 1:] + dx[:, :-1])
    return w.clamp_min(0.0)

class TimeDependentRFFKernel1D(nn.Module):
    """
    Time- & Space-Dependent RFF Kernel for MHLKNO.

    - 시간 t 에 따른 파라미터: 기존과 동일 (gain, length / GSM 파라미터)
    - 선택적 공간 의존성: x, t 를 입력으로 받는 작은 MLP 가
        각 head/basis 마다 주파수 ω를 위치별로 조정
      -> ω = ω_base(t) * (1 + δ(t, x)),  |δ| <= 0.5

    이때 φ(x) = [cos(ω(t,x) x), sin(ω(t,x) x)] 를 그대로 쓰므로
    K(x,y) = φ(x)^T φ(y) 는 항상 PSD이고, stationarity 가 사라져도 문제 없음.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        Q: int = 1,
        temb_dim: int = 256,
        t_hidden: int = 64,
        enable_time_cond: bool = True,
        num_heads: int = 16,
        num_basis: int = 100,
        bandwidth_init: float = 1.0,
        hyp_len_init: float | None = None,
        rff_type: str = "ghq",
        # --- NEW: 공간 의존 커널 옵션 ---
        enable_spatial_cond: bool = False,
        x_hidden: int = 64,
    ):
        super().__init__()
        self.kind = str(kernel_type).lower()
        self.rff_type = str(rff_type).lower()
        self.Q = int(Q)
        self.enable_time_cond = bool(enable_time_cond)
        self.num_heads = int(num_heads)
        self.num_basis = int(num_basis)

        # --- RFF/GHQ/QMC Sampling Initialization (기존) ---
        if self.rff_type == "ghq":
            points, weights = np.polynomial.hermite.hermgauss(self.num_basis)
            eps_fixed = torch.tensor(points, dtype=torch.float32).view(1, -1) * math.sqrt(2.0)
            w_tensor = torch.tensor(weights, dtype=torch.float32).view(1, -1) / math.sqrt(math.pi)
            self.register_buffer("eps", eps_fixed.repeat(self.num_heads, 1))        # (H, D)
            self.register_buffer("quad_weights", w_tensor.repeat(self.num_heads, 1))# (H, D)
        elif self.rff_type == "qmc":
            sobol = SobolEngine(dimension=1, scramble=True)
            eps_uniform = sobol.draw(self.num_heads * self.num_basis).view(self.num_heads, self.num_basis)
            eps = torch.distributions.Normal(0, 1).icdf(eps_uniform)
            self.register_buffer("eps", eps)
            self.register_buffer("quad_weights", torch.ones(self.num_heads, self.num_basis) / self.num_basis)
        else:
            eps = torch.randn(self.num_heads, self.num_basis)
            self.register_buffer("eps", eps)
            self.register_buffer("quad_weights", torch.ones(self.num_heads, self.num_basis) / self.num_basis)

        comp = torch.arange(self.num_basis) % max(self.Q, 1)
        self.register_buffer("basis_comp", comp)

        # --- Kernel Params (기존) ---
        if self.kind == "rbf":
            self.log_gain = nn.Parameter(torch.tensor(0.0))
            if hyp_len_init is None:
                hyp_len_init = 0.25
            self.log_len = nn.Parameter(torch.log(torch.tensor(float(hyp_len_init))))
            if self.enable_time_cond:
                self.tmlp = nn.Sequential(
                    nn.Linear(temb_dim, t_hidden),
                    nn.SiLU(),
                    nn.Linear(t_hidden, 2),
                )
                self._init_tmlp()
            else:
                self.tmlp = None

        else:  # "gsm" (stationary GSM)
            self.log_w = nn.Parameter(torch.zeros(self.Q))
            self.log_sig = nn.Parameter(torch.zeros(self.Q))
            self.log_mu = nn.Parameter(torch.zeros(self.Q))
            if self.enable_time_cond:
                self.tmlp = nn.Sequential(
                    nn.Linear(temb_dim, t_hidden),
                    nn.SiLU(),
                    nn.Linear(t_hidden, 3 * self.Q),
                )
                self._init_tmlp()
            else:
                self.tmlp = None

        # --- 공간 의존성 (x,t) → Δω(x,t) ---
        self.enable_spatial_cond = bool(enable_spatial_cond)
        if self.enable_spatial_cond:
            in_dim = 1 + temb_dim       # [x_norm, temb]
            self.x_mlp = nn.Sequential(
                nn.Linear(in_dim, x_hidden),
                nn.SiLU(),
                nn.Linear(x_hidden, self.num_heads * self.num_basis),
            )
            for i, m in enumerate(self.x_mlp):
                if isinstance(m, nn.Linear):
                    if i < len(self.x_mlp) - 1:
                        # 중간 레이어는 기존처럼 초기화
                        m.weight.data = default_init()(m.weight.data.shape)
                        nn.init.zeros_(m.bias)
                    else:
                        # 마지막 레이어는 완전 0 초기화 → delta ≡ 0 으로 시작
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
        else:
            self.x_mlp = None

    def _init_tmlp(self):
        for m in self.tmlp:
            if isinstance(m, nn.Linear):
                m.weight.data = default_init()(m.weight.data.shape)
                nn.init.zeros_(m.bias)

    # ----- 시간 t-의존 파라미터 (기존) -----
    def _rbf_params(self, temb):
        B = temb.size(0)
        if self.enable_time_cond and self.tmlp is not None:
            delta = 1.5 * torch.tanh(self.tmlp(temb))  # (B,2)
            gain = F.softplus(self.log_gain + delta[:, 0].view(B, 1))
            ell  = F.softplus(self.log_len  + delta[:, 1].view(B, 1))
        else:
            gain = F.softplus(self.log_gain).view(1, 1).expand(B, -1)
            ell  = F.softplus(self.log_len).view(1, 1).expand(B, -1)
        return gain, ell

    def _gsm_params(self, temb):
        B = temb.size(0)
        if self.enable_time_cond and self.tmlp is not None:
            delta = 1.5 * torch.tanh(self.tmlp(temb))       # (B,3Q)
            dw, ds, dm = torch.split(delta, [self.Q]*3, dim=-1)
            w   = F.softplus(self.log_w.view(1, -1)   + dw)
            sig = F.softplus(self.log_sig.view(1, -1) + ds)
            mu  = F.softplus(self.log_mu.view(1, -1)  + dm)
        else:
            w   = F.softplus(self.log_w).view(1, -1).expand(B, -1)
            sig = F.softplus(self.log_sig).view(1, -1).expand(B, -1)
            mu  = F.softplus(self.log_mu).view(1, -1).expand(B, -1)
        return w, sig, mu

    # ----- Stationary RFF (시간 의존만, x는 안 쓰는 버전) -----
    def make_omega_and_amp(self, temb, *, device=None, dtype=None):
        """
        temb : (B, C_temb)
        return:
          omega_base: (B, H, D)
          amp_base  : (B, H, D)
        (stationary in x; non-stationary in t)
        """
        B = temb.size(0)
        device = device or temb.device
        dtype  = dtype  or temb.dtype

        eps = self.eps.to(device=device, dtype=dtype)              # (H, D)
        quad_w = self.quad_weights.to(device=device, dtype=dtype)  # (H, D)
        H, D = eps.shape

        scale_factor = (quad_w * self.num_basis).sqrt().unsqueeze(0)  # (1, H, D)

        if self.kind == "rbf":
            gain, ell = self._rbf_params(temb)          # (B,1), (B,1)
            scale = 1.0 / (ell + 1e-8)                 # (B,1)
            omega = eps.unsqueeze(0) * scale.view(B, 1, 1)      # (B, H, D)
            amp   = gain.view(B, 1, 1).sqrt() * scale_factor     # (B, H, D)
            return omega, amp

        # GSM
        w, sig, mu = self._gsm_params(temb)            # (B, Q)
        comp = self.basis_comp.to(device)              # (D,)
        w_bd   = w[:, comp]    # (B, D)
        sig_bd = sig[:, comp]
        mu_bd  = mu[:, comp]

        omega = 2.0 * math.pi * mu_bd.unsqueeze(1) + sig_bd.unsqueeze(1) * eps.unsqueeze(0)  # (B, H, D)
        amp   = w_bd.unsqueeze(1).clamp_min(1e-8).sqrt() * scale_factor                      # (B, H, D)
        return omega, amp

    # ----- NEW: 공간 의존 ω(t,x) -----
    def make_omega_and_amp_spatial(
        self,
        x_coords: torch.Tensor,     # (B, N)  정규화 좌표 u ∈ [-1,1]
        temb: torch.Tensor,         # (B, C_temb)
        *,
        device=None,
        dtype=None,
    ):
        """
        return:
          omega_x: (B, N, H, D)  位置별 주파수
          amp_x  : (B, N, H, D)  (여기서는 amplitude는 위치별로 broadcast)
        """
        B, N = x_coords.shape
        device = device or x_coords.device
        dtype  = dtype  or x_coords.dtype

        omega_base, amp_base = self.make_omega_and_amp(temb, device=device, dtype=dtype)
        omega_base = omega_base.to(device=device, dtype=dtype)   # (B, H, D)
        amp_base   = amp_base.to(device=device, dtype=dtype)     # (B, H, D)

        if not self.enable_spatial_cond or self.x_mlp is None:
            # x-비의존: 단순 broadcast
            omega = omega_base.unsqueeze(1).expand(B, N, self.num_heads, self.num_basis)
            amp   = amp_base.unsqueeze(1).expand(B, N, self.num_heads, self.num_basis)
            return omega, amp

        # (x, t) → Δω(x,t) 를 예측하는 작은 MLP
        x_norm = x_coords.to(device=device, dtype=dtype).unsqueeze(-1)            # (B, N, 1)
        temb_exp = temb.to(device=device, dtype=dtype).unsqueeze(1).expand(B, N, -1)  # (B, N, C_temb)
        h = torch.cat([x_norm, temb_exp], dim=-1)     # (B, N, 1 + C_temb)
        h_flat = h.reshape(B * N, -1)                 # (B*N, 1 + C_temb)

        delta_flat = self.x_mlp(h_flat)               # (B*N, H*D)
        delta = delta_flat.view(B, N, self.num_heads, self.num_basis)  # (B, N, H, D)

        # modulation range 제한: |δ| ≤ 0.5
        delta = 0.5 * torch.tanh(delta)

        omega = omega_base.unsqueeze(1) * (1.0 + delta)   # (B, N, H, D)
        amp   = amp_base.unsqueeze(1).expand(B, N, self.num_heads, self.num_basis)
        return omega, amp

class LinearizedKNOLayer(nn.Module):
    """
    Improved Multi-Head Linearized KNO Layer.
    Fixes the bottleneck issue by maintaining head_dim.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 32,
        num_basis: int = 16,
        temb_dim: int = 256,
        bandwidth_init: float = 1.0,
        measure_scale: float = 1.0,
        *,
        kernel_type: str = "rbf",
        kernel_Q: int = 1,
        kernel_hidden: int = 64,
        enable_kernel_time_cond: bool = True,
        kernel_time_hidden: int = 64,
        hyp_len_init: float | None = None,
        rff_type: str = "ghq",
        bias: bool = True,
        enable_spatial_cond: bool = False,
        spatial_hidden: int = 64,        
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_basis = int(num_basis)
        self.measure_scale = float(measure_scale)
        
        # [ARCH FIX] Head Dimension Calculation
        if in_channels % num_heads != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})")
        self.head_dim = in_channels // num_heads

        # 1) Input/Output Projection
        # Instead of reducing to num_heads, we keep full dimension but split logically later
        self.in_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # 2) Time embedding
        self.tact = nn.SiLU()
        self.tproj = nn.Linear(temb_dim, in_channels)
        self.tproj.weight.data = default_init()(self.tproj.weight.data.shape)
        nn.init.zeros_(self.tproj.bias)

        # 3) Kernel Params
        self.kernel = TimeDependentRFFKernel1D(
            kernel_type=kernel_type,
            Q=kernel_Q,
            temb_dim=temb_dim,
            t_hidden=kernel_time_hidden,
            enable_time_cond=enable_kernel_time_cond,
            num_heads=self.num_heads,
            num_basis=self.num_basis,
            bandwidth_init=bandwidth_init,
            hyp_len_init=hyp_len_init,
            rff_type=rff_type,
            enable_spatial_cond=enable_spatial_cond,
            x_hidden=spatial_hidden,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1))
        else:
            self.bias = None

    def _get_phi(self, x_coords: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Generates RFF features Φ(x) for all positions.

        x_coords : (B, N)  정규화 좌표 u ∈ [-1,1]
        temb     : (B, C_temb)
        return   : (B, N, H, 2D)
        """
        B, N = x_coords.shape

        # omega_x : (B, N, H, D)
        # amp_x   : (B, N, H, D)  (현재는 amplitude는 위치별로 broadcast된 형태)
        omega_x, amp_x = self.kernel.make_omega_and_amp_spatial(
            x_coords, temb, device=x_coords.device, dtype=x_coords.dtype
        )

        # Phase: (B, N, H, D)
        phase = x_coords.view(B, N, 1, 1) * omega_x

        cos_part = torch.cos(phase)
        sin_part = torch.sin(phase)

        if amp_x is not None:
            # amp_x: (B, N, H, D)
            cos_part = cos_part * amp_x
            sin_part = sin_part * amp_x

        # Concatenate real/imag parts: (B, N, H, 2D)
        phi = torch.cat([cos_part, sin_part], dim=-1)

        # RFF normalization (1/sqrt(D))
        phi = phi * (self.num_basis ** -0.5)
        return phi

    def forward(self, x: torch.Tensor, x_coords: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        H = self.num_heads
        D_h = self.head_dim

        # (1) Time Shift & Projection
        shift = self.tproj(self.tact(temb))
        x = x + shift.unsqueeze(-1)
        
        v = self.in_proj(x) # (B, C, N)
        
        # Reshape to (B, H, Head_Dim, N) for Multi-Head operation
        v = v.view(B, H, D_h, N)

        # (2) Integration Weights
        w = _trapz_weights_1d(x_coords) * self.measure_scale # (B, N)
        
        # Weighted Values: v * w(x)
        # (B, H, D_h, N) * (B, 1, 1, N) -> (B, H, D_h, N)
        v_weighted = v * w.view(B, 1, 1, N)

        # (3) RFF Features
        # phi: (B, N, H, 2D)
        phi = self._get_phi(x_coords, temb) 

        # (4) Linearized Integral (Attention)
        # We want to compute: Y(x) = phi(x)^T * sum_y [ phi(y) * v(y) * w(y) ]
        
        # Step A: Accumulate Context
        # phi: (B, N, H, 2D) -> permute to (B, H, 2D, N)
        phi_T = phi.permute(0, 2, 3, 1) 
        
        # Context: (B, H, 2D, N) @ (B, H, N, D_h) -> (B, H, 2D, D_h)
        # Note: standard matrix mult over N
        context = torch.matmul(phi_T, v_weighted.permute(0, 1, 3, 2))
        
        # Step B: Project Back
        # Y_heads: (B, H, N, 2D) @ (B, H, 2D, D_h) -> (B, H, N, D_h)
        y_heads = torch.matmul(phi.permute(0, 2, 1, 3), context)
        
        # Reshape back: (B, H, N, D_h) -> (B, C, N)
        y_heads = y_heads.permute(0, 1, 3, 2).reshape(B, C, N)

        # (5) Output Projection
        out = self.out_proj(y_heads)
        if self.bias is not None:
            out = out + self.bias
            
        return out

class MHLKNO(nn.Module):
    """
    MHL-KNO Model Architecture with Corrected Multi-Head Logic.
    """
    def __init__(self, config):
        super().__init__()
        cfg = config.model

        self.hidden_channels = cfg.hidden_channels
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.lifting_channels = cfg.lifting_channels
        self.projection_channels = cfg.projection_channels
        self.n_layers = cfg.n_layers

        self.num_heads = int(getattr(cfg, 'num_kernel_heads', 4)) # Config default 4
        self.num_basis = int(getattr(cfg, 'num_kernel_basis', 100))
        self.bandwidth_init = float(getattr(cfg, 'kernel_bandwidth_init', 1.0))
        self.rff_type = str(getattr(cfg, 'kernel_rff_type', 'ghq')).lower()
        
        measure_scale = float(getattr(cfg, "measure_scale", 1.0))
        self.norm_type = getattr(cfg, "norm", "group_norm")
        self.preactivation = getattr(cfg, "preactivation", True)
        self.skip_type = getattr(cfg, "skip", "soft-gating")

        # KNO Params
        self.kernel_type = str(getattr(cfg, "kernel_type", "gsm")).lower()
        self.kernel_Q = int(getattr(cfg, "kernel_Q", 6))
        self.kernel_hidden = int(getattr(cfg, "kernel_hidden", 64))
        self.enable_kernel_time_cond = bool(getattr(cfg, "enable_kernel_time_cond", True))
        self.kernel_time_hidden = int(getattr(cfg, "kernel_time_hidden", 64))
        hyp_len_init = getattr(config.data, "hyp_len", 0.2)
        self.enable_spatial_kernel  = bool(getattr(cfg, "enable_spatial_kernel", False))
        self.spatial_kernel_hidden  = int(getattr(cfg, "spatial_kernel_hidden", 64))

        # Validate Heads
        if self.hidden_channels % self.num_heads != 0:
             raise ValueError(f"Hidden channels {self.hidden_channels} must be divisible by num_heads {self.num_heads}")

        # Time Embedding
        self.Dense = nn.ModuleList([
            nn.Linear(self.lifting_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        ])
        for layer in self.Dense:
            layer.weight.data = default_init()(layer.weight.data.shape)
            nn.init.zeros_(layer.bias)

        # Lifting & Projection
        self.lifting = Lifting(in_channels=self.in_channels, out_channels=self.hidden_channels, n_dim=1)
        self.projection = Projection(
            in_channels=self.hidden_channels, out_channels=self.out_channels,
            hidden_channels=self.projection_channels, n_dim=1, non_linearity=F.gelu,
        )

        # Layers
        self.layers = nn.ModuleList([
            LinearizedKNOLayer(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                num_heads=self.num_heads,
                num_basis=self.num_basis,
                temb_dim=self.lifting_channels,
                bandwidth_init=self.bandwidth_init,
                measure_scale=measure_scale,
                kernel_type=self.kernel_type,
                kernel_Q=self.kernel_Q,
                kernel_hidden=self.kernel_hidden,
                enable_kernel_time_cond=self.enable_kernel_time_cond,
                kernel_time_hidden=self.kernel_time_hidden,
                hyp_len_init=hyp_len_init,
                rff_type=self.rff_type,
                bias=True,
                enable_spatial_cond=self.enable_spatial_kernel,
                spatial_hidden=self.spatial_kernel_hidden,
            )
            for _ in range(self.n_layers)
        ])

        # Norms & Skips
        if self.norm_type == "group_norm":
            self.norms = nn.ModuleList([nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels) for _ in range(self.n_layers)])
        else: self.norms = None

        self.skips = nn.ModuleList([
            skip_connection(self.hidden_channels, self.hidden_channels, n_dim=1, type=self.skip_type)
            for _ in range(self.n_layers)
        ])

        self.mlps = nn.ModuleList([
            MLP(in_channels=self.hidden_channels, hidden_channels=int(round(self.hidden_channels * 4.0)),
                dropout=0.0, n_dim=1, temb_dim=self.hidden_channels)
            for _ in range(self.n_layers)
        ])

    # Runner Compatibility Methods (Empty for MHLKNO as it doesn't use NUFNO gating)
    def all_kappas(self): return []
    def all_baselines(self): return []
    def all_band_gates(self): return []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, N) -> [signal, coord_norm]
        x_coord_norm = x[:, -1, :]
        h = self.lifting(x)

        temb = get_timestep_embedding(t, self.lifting_channels)
        temb = self.Dense[1](F.silu(self.Dense[0](temb)))
        h = h + temb.unsqueeze(-1)

        for i in range(self.n_layers):
            if self.preactivation:
                h = F.silu(h)
                if self.norms: h = self.norms[i](h)
            
            # MHL-KNO Layer
            h_k = self.layers[i](h, x_coord_norm, temb)

            if not self.preactivation and self.norms:
                h_k = self.norms[i](h_k)
            
            h = h_k + self.skips[i](h)
            h = self.mlps[i](h, temb)

        y = self.projection(h).squeeze(1)
        return y
