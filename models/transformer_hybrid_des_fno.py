# models/transformer_hybrid_des_fno.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import get_timestep_embedding, Lifting, Projection, default_init
from .mlp import MLP, skip_connection
# 재사용: NUDFT 스펙트럴 블록 (밴드 게이팅·시간 게이트·가보르 창 포함)
from .nufno import NUSpectralConv1D


# ------------------------------
# 1) Distance-Encoded Self-Attn
# ------------------------------
class DESMHSA(nn.Module):
    """
    Distance-Encoded Self-Attention (연속 좌표 전용 MHSA)
    - 좌표가 불균등해도 d_ij = |x_i - x_j|로 연속 상대위치 바이어스를 구성
    - 바이어스:   α_h * exp(-d^2 / ℓ_h^2)  +  Σ_r [ a_{h,r} cos(ω_{h,r} d) + c_{h,r} sin(ω_{h,r} d) ]
    - 입력: tokens (B, N, C), coords (B, N)
    """
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_fourier: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_omega_max: float = 32.0,      # 좌표는 [-1,1] → d ∈ [0,2]; ω 최대값은 10~64 정도가 적절
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.n_fourier = n_fourier

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # --- DES 파라미터 (헤드별) ---
        # ω_{h,r} > 0
        # 초기값: [ω_min..ω_max] 지수 등분
        w_min, w_max = 1.0, float(init_omega_max)
        w_init = torch.logspace(math.log10(w_min), math.log10(w_max), steps=n_fourier)
        self.omega_raw = nn.Parameter(w_init[None, :].repeat(n_heads, 1))  # (H, R)
        # 선형 결합 계수
        self.a = nn.Parameter(torch.zeros(n_heads, n_fourier))   # cos 계수
        self.c = nn.Parameter(torch.zeros(n_heads, n_fourier))   # sin 계수
        # 국소성 커널 파라미터(양수)
        self.alpha_raw = nn.Parameter(torch.full((n_heads,), math.log(math.expm1(0.5))))   # softplus≈0.5
        self.ell_raw   = nn.Parameter(torch.full((n_heads,), math.log(math.expm1(0.15))))  # softplus≈0.15
        # 바이어스 전체 스케일
        self.bias_scale_raw = nn.Parameter(torch.zeros(n_heads))  # 처음엔 0 → 점진적으로 활성화

        # 초기화
        nn.init.xavier_uniform_(self.qkv.weight); nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _softplus(x):  # 수치 안전한 softplus
        return F.softplus(x) + 1e-12

    def _make_bias(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, N)  in [-1,1]
        return: bias (B, H, N, N)  — 어텐션 로짓에 더해짐 (가산적)
        """
        B, N = coords.shape
        d = (coords.unsqueeze(2) - coords.unsqueeze(1)).abs()  # (B, N, N)

        # 헤드 축으로 브로드캐스트
        dH = d.unsqueeze(1)  # (B, 1, N, N)

        # 국소 가우스 커널
        alpha = self._softplus(self.alpha_raw).view(1, self.n_heads, 1, 1)   # (1,H,1,1)
        ell   = self._softplus(self.ell_raw).view(1, self.n_heads, 1, 1)     # (1,H,1,1)
        local = alpha * torch.exp(-(dH ** 2) / (ell ** 2))

        # 사인/코사인 항
        omega = self._softplus(self.omega_raw)  # (H,R)
        # d:[B,1,N,N], ω:[H,R] → (B,H,N,N,R)
        phase = dH.unsqueeze(-1) * omega.view(1, self.n_heads, 1, 1, self.n_fourier)
        trig_c = torch.cos(phase)
        trig_s = torch.sin(phase)
        # 선형결합: a·cos + c·sin → (B,H,N,N)
        a = self.a.view(1, self.n_heads, 1, 1, self.n_fourier)
        c = self.c.view(1, self.n_heads, 1, 1, self.n_fourier)
        fourier = (a * trig_c + c * trig_s).sum(dim=-1)

        bias_scale = torch.tanh(self.bias_scale_raw).view(1, self.n_heads, 1, 1)  # (-1,1) 안정
        bias = bias_scale * (local + fourier)
        return bias  # (B,H,N,N)

    def forward(self, x_tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        x_tokens: (B, N, C)
        coords  : (B, N)  normalized coordinates [-1,1]
        """
        B, N, C = x_tokens.shape
        qkv = self.qkv(x_tokens)  # (B, N, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, H, N, Dh)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        des_bias = self._make_bias(coords)                     # (B,H,N,N)
        attn = F.softmax(attn_logits + des_bias, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                                         # (B,H,N, Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, C)   # (B,N,C)
        out = self.proj_drop(self.proj(out))                   # (B,N,C)
        return out


# ------------------------------------------
# 2) 하이브리드: NU-Spectral ⊕ DES-Transformer
# ------------------------------------------
class TransformerHybridDESFNO(nn.Module):
    """
    1D 하이브리드: NUFNO(스펙트럼) + Distance-Encoded Self-Attn(공간) 병렬 → 게이팅 융합
    - 입력 채널 2개 [signal, coord_norm]; 출력 (B,N)
    - spectral_blocks를 그대로 노출하여 기존 hilbert_loss_fn 스펙트럼 손실 연동
    """
    def __init__(self, config):
        super().__init__()
        cfg = config.model
        # 기본 하이퍼
        self.n_layers         = int(getattr(cfg, "n_layers", 4))
        self.hidden_channels  = int(getattr(cfg, "hidden_channels", 256))
        self.in_channels      = int(getattr(cfg, "in_channels", 2))
        self.out_channels     = int(getattr(cfg, "out_channels", 1))
        self.lifting_channels = int(getattr(cfg, "lifting_channels", 256))
        self.proj_channels    = int(getattr(cfg, "projection_channels", 256))
        self.norm_type        = str(getattr(cfg, "norm", "group_norm"))
        self.preactivation    = bool(getattr(cfg, "preactivation", True))
        self.skip_type        = str(getattr(cfg, "skip", "soft-gating"))
        self.n_modes          = cfg.n_modes[0] if isinstance(cfg.n_modes, (list,tuple)) else int(cfg.n_modes)

        # DES Transformer 하이퍼
        self.n_heads     = int(getattr(cfg, "des_heads", 8))
        self.n_fourier   = int(getattr(cfg, "des_n_fourier", 8))
        self.attn_drop   = float(getattr(cfg, "des_attn_drop", 0.0))
        self.proj_drop   = float(getattr(cfg, "des_proj_drop", 0.0))
        self.omega_max   = float(getattr(cfg, "des_omega_max", 32.0))

        # 시간 임베딩
        self.Dense = nn.ModuleList([
            nn.Linear(self.lifting_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels,  self.hidden_channels),
        ])
        for layer in self.Dense:
            layer.weight.data = default_init()(layer.weight.data.shape)
            nn.init.zeros_(layer.bias)

        # 입·출력
        self.lifting   = Lifting(in_channels=self.in_channels, out_channels=self.hidden_channels, n_dim=1)
        self.projection= Projection(in_channels=self.hidden_channels, out_channels=self.out_channels,
                                    hidden_channels=self.proj_channels, n_dim=1)

        # --- 분기 A: NU 스펙트럼 블록들(레이어별 κ) ---
        self.spectral_blocks = nn.ModuleList([
            NUSpectralConv1D(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_modes=self.n_modes,
                separable=True,
                bias=True,
                temb_dim=self.lifting_channels,
                # 아래 옵션들은 기존 YAML을 그대로 읽어옴
                kappa_warm_start_scale=float(getattr(cfg, "freq_scale_init", 0.3)),
                enable_band_gating=bool(getattr(cfg, "enable_band_gating", True)),
                band_split_fracs=tuple(getattr(cfg, "band_split_fracs", (0.4, 0.8))),
                enable_time_freq_gate=bool(getattr(cfg, "enable_time_freq_gate", True)),
                time_gate_tau0_frac=float(getattr(cfg, "time_gate_tau0_frac", 0.15)),
                time_gate_alpha_init=float(getattr(cfg, "time_gate_alpha_init", 8.0)),
                time_gate_hidden=int(getattr(cfg, "time_gate_hidden", 128)),
                enable_gabor_window=bool(getattr(cfg, "enable_gabor_window", False)),
                gabor_omega_init=float(getattr(cfg, "gabor_omega_init", 0.25)),
            )
            for _ in range(self.n_layers)
        ])

        # --- 분기 B: DES MHSA 레이어들 ---
        self.des_blocks = nn.ModuleList([
            DESMHSA(
                dim=self.hidden_channels, n_heads=self.n_heads,
                n_fourier=self.n_fourier, attn_drop=self.attn_drop, proj_drop=self.proj_drop,
                init_omega_max=self.omega_max
            )
            for _ in range(self.n_layers)
        ])

        # 융합 게이트(초기엔 스펙트럼 우세: bias=-2.2 → σ≈0.10)
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.hidden_channels*2, self.hidden_channels, kernel_size=1, bias=True),
                nn.SiLU(),
                nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1, bias=True),
            ) for _ in range(self.n_layers)
        ])
        for fg in self.fusion_gates:
            nn.init.zeros_(fg[-1].bias)  # 마지막 conv bias는 아래에서 레벨셋
        self._gate_bias = nn.Parameter(torch.tensor(-2.1972246))  # logit(0.10)

        # Norms, Skips, MLPs
        if self.norm_type is None:
            self.norms = None
        elif self.norm_type == "group_norm":
            self.norms = nn.ModuleList([nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels) for _ in range(self.n_layers)])
        elif self.norm_type == "instance_norm":
            self.norms = nn.ModuleList([nn.InstanceNorm1d(num_features=self.hidden_channels) for _ in range(self.n_layers)])
        else:
            raise ValueError(f"Unsupported norm: {self.norm_type}")

        self.skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, n_dim=1, type=self.skip_type)
                                    for _ in range(self.n_layers)])
        self.mlps  = nn.ModuleList([MLP(in_channels=self.hidden_channels,
                                        hidden_channels=int(round(self.hidden_channels*4.0)),
                                        dropout=0.0, n_dim=1, temb_dim=self.hidden_channels)
                                    for _ in range(self.n_layers)])

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, N)  [signal, coord_norm]
        t: (B,)
        return: (B, N)
        """
        B, Cin, N = x.shape
        x_coord_norm = x[:, -1, :]                         # (B,N)  이미 [-1,1] 정규화 채널
        z = math.pi * x_coord_norm                         # 위상(NUDFT용) — nufno와 동일

        # Lift + Timestep embedding
        h = self.lifting(x)                                # (B,C,N)
        temb = get_timestep_embedding(t, self.lifting_channels)
        temb = self.Dense[0](temb); temb = self.Dense[1](F.silu(temb))
        h = h + temb.unsqueeze(-1)                         # (B,C,N)

        for i in range(self.n_layers):
            if self.preactivation:
                h = self.act(h)
                if self.norms is not None:
                    h = self.norms[i](h)

            # --- 분기 A: NU 스펙트럼 ---
            y_spec = self.spectral_blocks[i](h, z, temb)  # (B,C,N)

            # --- 분기 B: DES-MHSA (토큰화: (B,N,C)) ---
            tokens = h.transpose(1, 2)                     # (B,N,C)
            y_attn = self.des_blocks[i](tokens, x_coord_norm).transpose(1, 2)  # (B,C,N)

            # --- 융합(채널별 게이트) ---
            fuse_in = torch.cat([y_spec, y_attn], dim=1)   # (B,2C,N)
            g = self.fusion_gates[i](fuse_in)              # (B,C,N)
            g = torch.sigmoid(g + self._gate_bias)         # 초기 ~0.10
            h_new = g * y_attn + (1.0 - g) * y_spec

            # 스킵 연결 + MLP
            h = h_new + self.skips[i](h)
            h = self.mlps[i](h, temb)

        y = self.projection(h).squeeze(1)                  # (B,N)
        return y
