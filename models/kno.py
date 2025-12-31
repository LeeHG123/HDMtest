# models/kno.py
# Kernel Neural Operator (KNO, 1D)
# - Full kernel integral operator (optionally multi-head)
# - Supports kernel_type: rbf | gsm | nsgsm
# - Time-conditioning supported (global hyper-MLP on kernel parameters)

import math
from typing import Optional

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


# -----------------------------
# Small utilities
# -----------------------------
def _inv_softplus(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse of softplus: softplus(x)=log(1+exp(x)) -> x=log(exp(y)-1)."""
    y = y.clamp_min(eps)
    return torch.log(torch.expm1(y))


def _trapz_weights_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Trapezoidal weights for a (sorted) 1D grid per batch.
    x: (B, N)
    return: (B, N)
    """
    B, N = x.shape
    if N == 1:
        return torch.ones_like(x)
    dx = (x[:, 1:] - x[:, :-1]).abs()
    w = torch.zeros_like(x)
    w[:, 0] = 0.5 * dx[:, 0]
    w[:, -1] = 0.5 * dx[:, -1]
    if N > 2:
        w[:, 1:-1] = 0.5 * (dx[:, 1:] + dx[:, :-1])
    return w.clamp_min(0.0)


class _Pos(nn.Module):
    """Stable positive mapping."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=1.0) + 1e-8


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FeatureNet1D(nn.Module):
    """
    Local (x-dependent) parameters for NS-GSM:
      x -> (w(x), sigma(x), mu(x)) for each mixture component
    """
    def __init__(self, hidden: int = 64, Q: int = 2):
        super().__init__()
        self.Q = int(Q)
        self.core = _MLP(in_dim=1, hidden=hidden, out_dim=3 * self.Q)
        self.pos = _Pos()

    def forward(self, x_bn1: torch.Tensor):
        """
        x_bn1: (B, N, 1)
        returns:
          w, sig, mu: each (B, N, Q), all positive
        """
        out = self.core(x_bn1)  # (B,N,3Q)
        Q = self.Q
        w, sig, mu = out[..., :Q], out[..., Q:2 * Q], out[..., 2 * Q:3 * Q]
        return self.pos(w), self.pos(sig), self.pos(mu)


class KNOScalarKernel1D(nn.Module):
    """
    Learnable 1D kernels:
      - RBF:
            k(x,y) = gain * exp( - (x-y)^2 / ell^2 )
      - GSM (stationary):
            sum_q w_q * exp(-0.5*(sig_q*(x-y))^2) * cos(2*pi*mu_q*(x-y))
      - NS-GSM (nonstationary):
            x-dependent params via FeatureNet + Gibbs form
    Optional time-conditioning:
      - outputs deltas in log-parameter space (or gates for NS-GSM)
    """
    def __init__(
        self,
        *,
        kernel_type: str = "gsm",
        Q: int = 2,
        x_hidden: int = 64,
        use_time_cond: bool = True,
        temb_dim: int = 256,
        t_hidden: int = 64,
        rbf_len_init: Optional[float] = None,
    ):
        super().__init__()
        self.kind = str(kernel_type).lower()
        self.Q = int(Q)
        self.pos = _Pos()

        # base static params
        if self.kind == "rbf":
            self.log_gain = nn.Parameter(torch.tensor(0.0))
            ell0 = float(rbf_len_init) if rbf_len_init is not None else 0.25
            self.log_len = nn.Parameter(_inv_softplus(torch.tensor(ell0, dtype=torch.float32)))
        elif self.kind == "gsm":
            self.log_w = nn.Parameter(torch.zeros(self.Q))
            self.log_sig = nn.Parameter(torch.zeros(self.Q))
            self.log_mu = nn.Parameter(torch.zeros(self.Q))
        elif self.kind == "nsgsm":
            self.feat = _FeatureNet1D(hidden=x_hidden, Q=self.Q)
        else:
            raise ValueError(f"unknown kernel_type={kernel_type}")

        # (optional) time-conditioning
        self.use_time_cond = bool(use_time_cond)
        if self.use_time_cond:
            if self.kind == "rbf":
                self.tmlp = _MLP(temb_dim, t_hidden, 2)        # d(log_gain), d(log_len)
            elif self.kind == "gsm":
                self.tmlp = _MLP(temb_dim, t_hidden, 3 * self.Q)  # d(log_w), d(log_sig), d(log_mu)
            else:  # nsgsm
                self.tmlp = _MLP(temb_dim, t_hidden, 2 * self.Q)  # gates for w,sigma per component
        else:
            self.tmlp = None

    def forward(self, x: torch.Tensor, y: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x, y  : (B, N) normalized coords in [-1,1]
        temb  : (B, C_temb) or None
        return: (B, N, N)
        """
        B, N = x.shape
        x_ = x.unsqueeze(2)  # (B,N,1)
        y_ = y.unsqueeze(1)  # (B,1,N)

        if self.kind == "rbf":
            if self.use_time_cond and (temb is not None):
                delta = 1.5 * torch.tanh(self.tmlp(temb))   # (B,2)
                d_gain, d_len = delta[:, 0], delta[:, 1]
                gain = self.pos(self.log_gain + d_gain.view(B, 1))  # (B,1)
                ell = self.pos(self.log_len + d_len.view(B, 1))     # (B,1)
            else:
                gain = self.pos(self.log_gain).view(1, 1)
                ell = self.pos(self.log_len).view(1, 1)

            diff2 = (x_ - y_) ** 2  # (B,N,N)
            K = gain.view(B if gain.shape[0] == B else 1, 1, 1) * torch.exp(
                -diff2 / (ell.view(B if ell.shape[0] == B else 1, 1, 1) ** 2)
            )
            if K.shape[0] == 1:
                K = K.expand(B, N, N)
            return K

        if self.kind == "gsm":
            if self.use_time_cond and (temb is not None):
                delta = 1.5 * torch.tanh(self.tmlp(temb))  # (B,3Q)
                dw, ds, dm = torch.split(delta, [self.Q, self.Q, self.Q], dim=-1)
                w = self.pos(self.log_w.view(1, -1) + dw)         # (B,Q)
                sig = self.pos(self.log_sig.view(1, -1) + ds)     # (B,Q)
                mu = self.pos(self.log_mu.view(1, -1) + dm)       # (B,Q)
            else:
                w = self.pos(self.log_w).view(1, self.Q).expand(B, -1)
                sig = self.pos(self.log_sig).view(1, self.Q).expand(B, -1)
                mu = self.pos(self.log_mu).view(1, self.Q).expand(B, -1)

            diff = (x_ - y_).unsqueeze(-1)  # (B,N,N,1)
            gaus = torch.exp(-0.5 * (sig.view(B, 1, 1, self.Q) * diff).pow(2))          # (B,N,N,Q)
            osc = torch.cos(2 * math.pi * (mu.view(B, 1, 1, self.Q) * diff))            # (B,N,N,Q)
            K = (w.view(B, 1, 1, self.Q) * gaus * osc).sum(dim=-1)                       # (B,N,N)
            return K

        # NS-GSM (nonstationary)
        wx, sx, mux = self.feat(x.unsqueeze(-1))  # (B,N,Q)
        wy, sy, muy = self.feat(y.unsqueeze(-1))  # (B,N,Q)

        if self.use_time_cond and (temb is not None):
            gate = torch.tanh(self.tmlp(temb))     # (B,2Q) in [-1,1]
            gw, gs = gate[:, :self.Q], gate[:, self.Q:]
            wx = wx * (1 + 0.25 * gw.unsqueeze(1))
            sx = sx * (1 + 0.25 * gs.unsqueeze(1))
            wy = wy * (1 + 0.25 * gw.unsqueeze(1))
            sy = sy * (1 + 0.25 * gs.unsqueeze(1))

        # broadcast
        wx = wx.unsqueeze(2)    # (B,N,1,Q)
        wy = wy.unsqueeze(1)    # (B,1,N,Q)
        sx = sx.unsqueeze(2)    # (B,N,1,Q)
        sy = sy.unsqueeze(1)    # (B,1,N,Q)
        mux = mux.unsqueeze(2)  # (B,N,1,Q)
        muy = muy.unsqueeze(1)  # (B,1,N,Q)

        r = sx * sx + sy * sy
        dx2 = (x_ - y_) ** 2
        r_eps = r + 1e-8
        gibbs = torch.sqrt(2 * sx * sy / r_eps) * torch.exp(-dx2.unsqueeze(-1) / r_eps)
        phase = torch.cos(2 * math.pi * (mux * x_.unsqueeze(-1) - muy * y_.unsqueeze(-1)))
        K = (wx * wy * gibbs * phase).sum(dim=-1)
        return K


class KNOSpectralIntegral1D(nn.Module):
    """
    Single-head full-kernel integral block:
      y = K @ (w ⊙ f) + PW(f)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temb_dim: int = 256,
        kernel_type: str = "gsm",
        kernel_Q: int = 2,
        kernel_x_hidden: int = 64,
        kernel_time_hidden: int = 64,
        enable_kernel_time_cond: bool = True,
        bias: bool = True,
        measure_scale: float = 1.0,
        rbf_len_init: Optional[float] = None,
    ):
        super().__init__()
        self.kern = KNOScalarKernel1D(
            kernel_type=kernel_type,
            Q=kernel_Q,
            x_hidden=kernel_x_hidden,
            use_time_cond=enable_kernel_time_cond,
            temb_dim=temb_dim,
            t_hidden=kernel_time_hidden,
            rbf_len_init=rbf_len_init,
        )
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1)) if bias else None
        self.measure_scale = float(measure_scale)

        # time-to-channel shift
        self.tact = nn.SiLU()
        self.tproj = nn.Linear(temb_dim, in_channels)
        self.tproj.weight.data = default_init()(self.tproj.weight.data.shape)
        nn.init.zeros_(self.tproj.bias)

    def forward(self, x_feat: torch.Tensor, z: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x_feat: (B, C_in, N)
        z     : (B, N) where z = pi * x_norm
        temb  : (B, C_temb)
        """
        B, C, N = x_feat.shape

        if temb is not None:
            shift = self.tproj(self.tact(temb))     # (B,C)
            x_feat = x_feat + shift.unsqueeze(-1)

        x_norm = z / math.pi                        # (B,N) in [-1,1]
        w = _trapz_weights_1d(x_norm) * self.measure_scale
        K = self.kern(x_norm, x_norm, temb)         # (B,N,N)

        xw = x_feat * w.unsqueeze(1)                # (B,C,N)
        y = torch.einsum("bmn,bcn->bcm", K, xw)     # (B,C,N)

        y = y + self.pw(x_feat)                     # channel mixing
        if self.bias is not None:
            y = y + self.bias
        return y


class MultiHeadKNOSpectralIntegral1D(nn.Module):
    """
    Multi-head full-kernel integral block.
    Each head can have an independent kernel (share_kernel_across_heads=False).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temb_dim: int = 256,
        kernel_type: str = "gsm",
        kernel_Q: int = 2,
        kernel_x_hidden: int = 64,
        kernel_time_hidden: int = 64,
        enable_kernel_time_cond: bool = True,
        bias: bool = True,
        measure_scale: float = 1.0,
        num_heads: int = 4,
        share_kernel_across_heads: bool = False,
        rbf_len_init: Optional[float] = None,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)
        self.measure_scale = float(measure_scale)
        self.share_kernel = bool(share_kernel_across_heads)

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.in_channels % self.num_heads != 0:
            raise ValueError(
                f"in_channels ({self.in_channels}) must be divisible by num_heads ({self.num_heads})"
            )
        self.head_dim = self.in_channels // self.num_heads

        # time embedding -> channel shift
        self.tact = nn.SiLU()
        self.tproj = nn.Linear(temb_dim, self.in_channels)
        self.tproj.weight.data = default_init()(self.tproj.weight.data.shape)
        nn.init.zeros_(self.tproj.bias)

        # channel projections
        self.in_proj = nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1)
        self.out_proj = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)

        # kernels
        if self.share_kernel:
            self.kern = KNOScalarKernel1D(
                kernel_type=kernel_type,
                Q=kernel_Q,
                x_hidden=kernel_x_hidden,
                use_time_cond=enable_kernel_time_cond,
                temb_dim=temb_dim,
                t_hidden=kernel_time_hidden,
                rbf_len_init=rbf_len_init,
            )
        else:
            self.kern = nn.ModuleList([
                KNOScalarKernel1D(
                    kernel_type=kernel_type,
                    Q=kernel_Q,
                    x_hidden=kernel_x_hidden,
                    use_time_cond=enable_kernel_time_cond,
                    temb_dim=temb_dim,
                    t_hidden=kernel_time_hidden,
                    rbf_len_init=rbf_len_init,
                )
                for _ in range(self.num_heads)
            ])

        self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1)) if bias else None

    def _get_kernel(self, h: int) -> KNOScalarKernel1D:
        if isinstance(self.kern, nn.ModuleList):
            return self.kern[h]
        return self.kern

    def forward(self, x_feat: torch.Tensor, z: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        x_feat : (B, C_in, N)
        z      : (B, N) where z = pi * x_norm
        temb   : (B, C_temb)
        return : (B, C_out, N)
        """
        B, C, N = x_feat.shape

        # time shift
        shift = self.tproj(self.tact(temb))           # (B, C_in)
        x_feat = x_feat + shift.unsqueeze(-1)

        # value projection + head reshape
        v = self.in_proj(x_feat)                      # (B, C_in, N)
        v = v.view(B, self.num_heads, self.head_dim, N)

        # coords & trapz weights
        x_norm = z / math.pi
        w = _trapz_weights_1d(x_norm) * self.measure_scale
        v_weighted = v * w.view(B, 1, 1, N)

        # head-wise integrals
        y_heads = torch.zeros_like(v_weighted)        # (B, H, D_h, N)
        for h in range(self.num_heads):
            kern_h = self._get_kernel(h)
            K = kern_h(x_norm, x_norm, temb)          # (B, N, N)
            v_h = v_weighted[:, h, :, :]              # (B, D_h, N)
            y_h = torch.einsum("bmn,bcn->bcm", K, v_h)  # (B, D_h, N)
            y_heads[:, h, :, :] = y_h

        # concat + out proj
        y_cat = y_heads.view(B, C, N)
        out = self.out_proj(y_cat)
        if self.bias is not None:
            out = out + self.bias
        return out


class KNO(nn.Module):
    """
    1D Kernel-Integral Operator (KNO) model.

    Input channels: 2 = [signal, coord_norm]
    Keeps the same outer skeleton as your other 1D models:
      - time embedding
      - GroupNorm
      - soft-gating skip
      - MLP
    """
    def __init__(self, config):
        super().__init__()
        cfg = config.model

        self.n_dim = 1
        self.n_modes = cfg.n_modes
        assert isinstance(self.n_modes, (list, tuple)) and len(self.n_modes) == 1, \
            "KNO is 1D only: n_modes must be [int,]."

        self.hidden_channels = cfg.hidden_channels
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.lifting_channels = cfg.lifting_channels
        self.projection_channels = cfg.projection_channels
        self.n_layers = cfg.n_layers
        self.norm_type = getattr(cfg, "norm", "group_norm")
        self.preactivation = getattr(cfg, "preactivation", True)
        self.skip_type = getattr(cfg, "skip", "soft-gating")

        # multi-head option
        self.num_heads = int(getattr(cfg, "num_kernel_heads", 1))
        if self.num_heads < 1:
            raise ValueError(f"num_kernel_heads must be >= 1, got {self.num_heads}")
        if self.hidden_channels % self.num_heads != 0:
            raise ValueError(
                f"hidden_channels ({self.hidden_channels}) must be divisible by num_kernel_heads ({self.num_heads})"
            )

        # time embedding
        self.Dense = nn.ModuleList([
            nn.Linear(self.lifting_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        ])
        for layer in self.Dense:
            layer.weight.data = default_init()(layer.weight.data.shape)
            nn.init.zeros_(layer.bias)

        # lifting/projection
        self.lifting = Lifting(in_channels=self.in_channels, out_channels=self.hidden_channels, n_dim=1)
        self.projection = Projection(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_dim=1,
            non_linearity=F.gelu,
        )

        # measure scale (Jacobian correction injected by runner)
        measure_scale = float(getattr(cfg, "measure_scale", 1.0))

        # kernel hyper
        kno_type = str(getattr(cfg, "kernel_type", "gsm")).lower()          # rbf|gsm|nsgsm
        kno_Q = int(getattr(cfg, "kernel_Q", 2))
        kno_hidden = int(getattr(cfg, "kernel_hidden", 64))
        kno_t_hidden = int(getattr(cfg, "kernel_time_hidden", 64))
        kno_t_cond = bool(getattr(cfg, "enable_kernel_time_cond", True))
        rbf_len_init = float(getattr(config.data, "hyp_len", 0.25))

        # spectral blocks (single-head vs multi-head)
        if self.num_heads == 1:
            self.spectral_blocks = nn.ModuleList([
                KNOSpectralIntegral1D(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    temb_dim=self.lifting_channels,
                    kernel_type=kno_type,
                    kernel_Q=kno_Q,
                    kernel_x_hidden=kno_hidden,
                    kernel_time_hidden=kno_t_hidden,
                    enable_kernel_time_cond=kno_t_cond,
                    bias=True,
                    measure_scale=measure_scale,
                    rbf_len_init=rbf_len_init,
                )
                for _ in range(self.n_layers)
            ])
        else:
            self.spectral_blocks = nn.ModuleList([
                MultiHeadKNOSpectralIntegral1D(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    temb_dim=self.lifting_channels,
                    kernel_type=kno_type,
                    kernel_Q=kno_Q,
                    kernel_x_hidden=kno_hidden,
                    kernel_time_hidden=kno_t_hidden,
                    enable_kernel_time_cond=kno_t_cond,
                    bias=True,
                    measure_scale=measure_scale,
                    num_heads=self.num_heads,
                    share_kernel_across_heads=False,
                    rbf_len_init=rbf_len_init,
                )
                for _ in range(self.n_layers)
            ])

        # norms
        if self.norm_type is None:
            self.norms = None
        elif self.norm_type == "group_norm":
            self.norms = nn.ModuleList([
                nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels)
                for _ in range(self.n_layers)
            ])
        elif self.norm_type == "instance_norm":
            self.norms = nn.ModuleList([
                nn.InstanceNorm1d(num_features=self.hidden_channels)
                for _ in range(self.n_layers)
            ])
        else:
            raise ValueError(f"Unsupported norm: {self.norm_type}")

        # skips / mlps
        self.skips = nn.ModuleList([
            skip_connection(self.hidden_channels, self.hidden_channels, n_dim=1, type=self.skip_type)
            for _ in range(self.n_layers)
        ])

        self.use_mlp = True
        self.mlps = nn.ModuleList([
            MLP(
                in_channels=self.hidden_channels,
                hidden_channels=int(round(self.hidden_channels * 4.0)),
                dropout=0.0,
                n_dim=1,
                temb_dim=self.hidden_channels,
            )
            for _ in range(self.n_layers)
        ]) if self.use_mlp else None

    # Runner compatibility (NUFNO hooks are absent)
    def all_kappas(self):      return []
    def all_baselines(self):   return []
    def all_band_gates(self):  return []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 2, N) = [signal, coord_norm]
        t : (B,)
        return: (B, N)
        """
        x_coord_norm = x[:, -1, :]  # (B,N)

        # lifting
        h = self.lifting(x)  # (B, C_hid, N)

        # time embedding
        temb = get_timestep_embedding(t, self.lifting_channels)
        temb = self.Dense[0](temb)
        temb = self.Dense[1](F.silu(temb))
        h = h + temb.unsqueeze(-1)

        # phase coordinate
        z = math.pi * x_coord_norm

        for i in range(self.n_layers):
            if self.preactivation:
                h = F.silu(h)
                if self.norms is not None:
                    h = self.norms[i](h)

            h_f = self.spectral_blocks[i](h, z, temb)

            if (not self.preactivation) and (self.norms is not None):
                h_f = self.norms[i](h_f)

            h = h_f + self.skips[i](h)

            if self.use_mlp:
                h = self.mlps[i](h, temb)

        y = self.projection(h).squeeze(1)
        return y






