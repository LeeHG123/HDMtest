import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fno import Lifting, Projection, get_timestep_embedding, default_init
from .mlp import MLP, skip_connection

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _trapz_weights_1d(x: torch.Tensor) -> torch.Tensor:
    """Trapezoidal integration weights for 1D grid."""
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

class TimeDependentChebyshevKernel1D(nn.Module):
    """
    Deterministically Decomposed GSM Kernel via Cotter's Method (Taylor Expansion).
    
    Approximates the kernel:
        k(x, y) = sum_q w_q * exp(-0.5 * sigma_q^2 * (x-y)^2) * cos(2pi * mu_q * (x-y))
    
    Using explicit Taylor expansion features phi(x) such that:
        k(x, y) approx <phi(x), phi(y)>
        
    Unlike RFF, this is fully deterministic and uses polynomial basis.
    To ensure Multi-Head diversity without random noise, we make GSM parameters (w, sigma, mu)
    independent per head.
    """

    def __init__(
        self,
        taylor_degree: int = 10,     # Taylor series truncation order (D)
        kernel_Q: int = 6,           # Number of GSM components (Q)
        num_heads: int = 4,          # Number of heads (H)
        temb_dim: int = 256,
        t_hidden: int = 64,
        enable_time_cond: bool = True,
        # Initializers
        sigma_init_scale: float = 1.0, 
    ):
        super().__init__()
        self.D = taylor_degree
        self.Q = kernel_Q
        self.H = num_heads
        self.enable_time_cond = enable_time_cond

        # 1. Learnable GSM Parameters per Head
        # Shape: (H, Q) to allow each head to capture different spectral patterns
        # log_w: mixture weights
        self.log_w = nn.Parameter(torch.zeros(self.H, self.Q))
        
        # log_sig: bandwidths (inverse length scale)
        # Initialize with some variance to cover different frequencies
        self.log_sig = nn.Parameter(torch.randn(self.H, self.Q) * 0.5 + math.log(sigma_init_scale))
        
        # log_mu: center frequencies
        # Initialize uniformly to cover spectrum
        self.log_mu = nn.Parameter(torch.randn(self.H, self.Q))

        # 2. Time Conditioning MLP
        # Output delta for all heads and components: H * 3 * Q parameters
        if self.enable_time_cond:
            self.output_dim = self.H * 3 * self.Q
            self.tmlp = nn.Sequential(
                nn.Linear(temb_dim, t_hidden),
                nn.SiLU(),
                nn.Linear(t_hidden, self.output_dim),
            )
            self._init_tmlp()
        else:
            self.tmlp = None

        # 3. Precompute Factorials for Taylor Coefficients
        # log(n!) for n = 0 to D-1
        factorials = torch.tensor([math.factorial(i) for i in range(self.D)], dtype=torch.float32)
        self.register_buffer("log_factorial", torch.log(factorials)) # (D,)

        # Feature dimension per head = 2 (cos/sin) * Q (components) * D (degree)
        self.feature_dim = 2 * self.Q * self.D

    def _init_tmlp(self):
        for m in self.tmlp:
            if isinstance(m, nn.Linear):
                m.weight.data = default_init()(m.weight.data.shape)
                nn.init.zeros_(m.bias)

    def _get_gsm_params(self, temb):
        """
        Returns w, sigma, mu for current batch and time.
        Output shape: (B, H, Q)
        """
        B = temb.size(0)
        
        # Base parameters: (1, H, Q)
        base_w   = self.log_w.unsqueeze(0)
        base_sig = self.log_sig.unsqueeze(0)
        base_mu  = self.log_mu.unsqueeze(0)

        if self.enable_time_cond and self.tmlp is not None:
            # delta: (B, H * 3 * Q)
            delta = 1.5 * torch.tanh(self.tmlp(temb)) 
            delta = delta.view(B, self.H, 3 * self.Q)
            
            dw = delta[..., :self.Q]
            ds = delta[..., self.Q : 2*self.Q]
            dm = delta[..., 2*self.Q :]
            
            w   = F.softplus(base_w + dw)
            sig = F.softplus(base_sig + ds)
            mu  = F.softplus(base_mu + dm)
        else:
            w   = F.softplus(base_w).expand(B, -1, -1)
            sig = F.softplus(base_sig).expand(B, -1, -1)
            mu  = F.softplus(base_mu).expand(B, -1, -1)
            
        return w, sig, mu

    def get_feature_map(self, x_coords: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Computes the Explicit Taylor Feature Map phi(x).
        
        Args:
            x_coords: (B, N) normalized coordinates in [-1, 1]
            temb: (B, C_temb) time embedding
            
        Returns:
            phi: (B, N, H, FeatureDim)
        """
        B, N = x_coords.shape
        device = x_coords.device
        
        # 1. Get GSM Parameters: (B, H, Q)
        w, sig, mu = self._get_gsm_params(temb)
        
        # 2. Prepare Tensors for Broadcasting
        # x: (B, N, 1, 1, 1) to align with (B, 1, H, Q, D)
        x = x_coords.view(B, N, 1, 1, 1)
        
        # Params: (B, 1, H, Q, 1)
        w   = w.view(B, 1, self.H, self.Q, 1)
        sig = sig.view(B, 1, self.H, self.Q, 1)
        mu  = mu.view(B, 1, self.H, self.Q, 1)
        
        # Order n: (1, 1, 1, 1, D)
        n_range = torch.arange(self.D, device=device, dtype=torch.float32).view(1, 1, 1, 1, self.D)
        log_fact = self.log_factorial.view(1, 1, 1, 1, self.D)

        # 3. Compute Taylor Coefficients (Gaussian Envelope)
        # We compute in log-space for stability:
        # log_coeff = 0.5 * ( log(w) - sig^2 * x^2 + n * log(sig^2) - log(n!) )
        # term      = exp(log_coeff) * x^n
        
        sig_sq = sig.pow(2)
        log_w = torch.log(w + 1e-8)
        log_sig_sq = torch.log(sig_sq + 1e-8)
        
        # Common Gaussian decay part: -0.5 * sig^2 * x^2
        decay = -0.5 * sig_sq * x.pow(2)
        
        # Polynomial part coefficient
        poly_coeff_log = 0.5 * (log_w + n_range * log_sig_sq - log_fact)
        
        # Combine
        log_total_coeff = decay + poly_coeff_log
        
        # envelope = exp(log_total_coeff) * x^n
        # Note: x^n can be computed as exp(n * log|x|) * sign(x)^n, but pow is safe here
        envelope = torch.exp(log_total_coeff) * x.pow(n_range) # (B, N, H, Q, D)
        
        # 4. Compute Trigonometric Part (Periodic)
        # arg = 2 * pi * mu * x
        # cos_part = cos(arg), sin_part = sin(arg)
        # Note: These broadcast over D axis
        arg = 2 * math.pi * mu * x
        cos_part = torch.cos(arg) # (B, N, H, Q, 1)
        sin_part = torch.sin(arg) # (B, N, H, Q, 1)
        
        # 5. Construct Full Feature Map
        # phi_cos = envelope * cos_part
        # phi_sin = envelope * sin_part
        phi_cos = envelope * cos_part
        phi_sin = envelope * sin_part
        
        # Concatenate Real and Imaginary parts
        # Shape: (B, N, H, Q, 2*D) -> Flatten last dims -> (B, N, H, 2*Q*D)
        phi = torch.cat([phi_cos, phi_sin], dim=-1)
        phi = phi.view(B, N, self.H, -1)
        
        return phi

class LinearizedChebyshevKNOLayer(nn.Module):
    """
    Linearized KNO Layer utilizing Deterministic Chebyshev/Taylor features.
    Replaces RFF sampling with explicit polynomial expansion.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        taylor_degree: int = 10,
        kernel_Q: int = 6,
        temb_dim: int = 256,
        measure_scale: float = 1.0,
        kernel_time_hidden: int = 64,
        enable_kernel_time_cond: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.measure_scale = float(measure_scale)
        
        if in_channels % num_heads != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})")
        self.head_dim = in_channels // num_heads

        # 1. Projections
        self.in_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # 2. Time embedding projection
        self.tact = nn.SiLU()
        self.tproj = nn.Linear(temb_dim, in_channels)
        self.tproj.weight.data = default_init()(self.tproj.weight.data.shape)
        nn.init.zeros_(self.tproj.bias)

        # 3. Deterministic Kernel (Cotter's Method)
        self.kernel = TimeDependentChebyshevKernel1D(
            taylor_degree=taylor_degree,
            kernel_Q=kernel_Q,
            num_heads=num_heads,
            temb_dim=temb_dim,
            t_hidden=kernel_time_hidden,
            enable_time_cond=enable_kernel_time_cond,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, x_coords: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        H = self.num_heads
        D_h = self.head_dim

        # (1) Time Shift & Projection
        shift = self.tproj(self.tact(temb))
        x = x + shift.unsqueeze(-1)
        
        v = self.in_proj(x) # (B, C, N)
        v = v.view(B, H, D_h, N)

        # (2) Integration Weights
        w = _trapz_weights_1d(x_coords) * self.measure_scale # (B, N)
        v_weighted = v * w.view(B, 1, 1, N) # (B, H, D_h, N)

        # (3) Explicit Taylor Features
        # phi: (B, N, H, FeatureDim)
        phi = self.kernel.get_feature_map(x_coords, temb)
        
        # (4) Linearized Integral (Attention)
        # K(x,y) = phi(x)^T phi(y)
        # Integral = phi(x)^T * sum_y [ phi(y) * v(y) * w(y) ]
        
        # Step A: Accumulate Context (Global Descriptor)
        # phi: (B, N, H, F) -> permute to (B, H, F, N)
        phi_T = phi.permute(0, 2, 3, 1)
        
        # Context: (B, H, F, N) @ (B, H, N, D_h) -> (B, H, F, D_h)
        # This is the O(N) step
        context = torch.matmul(phi_T, v_weighted.permute(0, 1, 3, 2))
        
        # Step B: Project Back (Evaluate at x)
        # Y_heads: (B, H, N, F) @ (B, H, F, D_h) -> (B, H, N, D_h)
        # Note: phi was (B, N, H, F), so permute to (B, H, N, F)
        y_heads = torch.matmul(phi.permute(0, 2, 1, 3), context)
        
        # Reshape back: (B, H, N, D_h) -> (B, C, N)
        y_heads = y_heads.permute(0, 1, 3, 2).reshape(B, C, N)

        # (5) Output Projection
        out = self.out_proj(y_heads)
        if self.bias is not None:
            out = out + self.bias
            
        return out

class ChebyshevMHLKNO(nn.Module):
    """
    Multi-Head Linearized KNO with Deterministic Chebyshev/Taylor Approximation.
    Replaces Bochner/RFF with Cotter's Method for stable, low-rank kernel approximation
    in bounded domains [-1, 1].
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

        # Chebyshev/Taylor Specific Configs
        # Default num_heads=4, taylor_degree=10, kernel_Q=6
        self.num_heads = int(getattr(cfg, 'num_kernel_heads', 4))
        self.taylor_degree = int(getattr(cfg, 'taylor_degree', 12)) # D
        self.kernel_Q = int(getattr(cfg, 'kernel_Q', 6))            # Q
        
        self.kernel_time_hidden = int(getattr(cfg, 'kernel_time_hidden', 64))
        self.enable_kernel_time_cond = bool(getattr(cfg, 'enable_kernel_time_cond', True))
        measure_scale = float(getattr(cfg, "measure_scale", 1.0))

        self.norm_type = getattr(cfg, "norm", "group_norm")
        self.preactivation = getattr(cfg, "preactivation", True)
        self.skip_type = getattr(cfg, "skip", "soft-gating")

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
            LinearizedChebyshevKNOLayer(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                num_heads=self.num_heads,
                taylor_degree=self.taylor_degree,
                kernel_Q=self.kernel_Q,
                temb_dim=self.lifting_channels,
                measure_scale=measure_scale,
                kernel_time_hidden=self.kernel_time_hidden,
                enable_kernel_time_cond=self.enable_kernel_time_cond,
                bias=True,
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

    # Methods for Runner compatibility (empty lists)
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
            
            # Chebyshev Linearized KNO Layer
            h_k = self.layers[i](h, x_coord_norm, temb)

            if not self.preactivation and self.norms:
                h_k = self.norms[i](h_k)
            
            h = h_k + self.skips[i](h)
            h = self.mlps[i](h, temb)

        y = self.projection(h).squeeze(1)
        return y