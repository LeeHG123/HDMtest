# models/nufno.py
# Non-uniform Fourier Neural Operator (NUFNO) for 1D
# - 좌표가 균등 격자가 아니어도 동작
# - 기존 FNO와 동일한 인터페이스 (forward(x, t) -> (B, N))
# - 입력 x는 (B, C_in, N)이며, 마지막 채널은 좌표 채널(x_coord)라고 가정
# - 시간 임베딩, GroupNorm, Soft-gating skip, (선택) MLP를 그대로 재사용

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


def _build_modes(n_modes: int) -> torch.Tensor:
    """
    n_modes: 정수(예: 64). 실제로는 [-K..K]로 총 M = 2*K+1개 모드 사용.
    """
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")
    K = n_modes // 2
    ks = torch.arange(-K, K + 1, dtype=torch.float32)  # [-K, ..., 0, ..., K]
    return ks  # (M,)


class NUSpectralConv1D(nn.Module):
    """
    1D Non-uniform Spectral Convolution (NUFFT 근사)
    - 분해: NUFFT(forward) → W(복소 가중치) → NUFFT(inverse)
    - separable=True 일 때 채널별(depthwise)로 모드마다 가중(빠르고 메모리 적음)
    - separable=False 일 때 (out_ch, in_ch, mode) 복소행렬로 완전 연결

    Shapes
    ------
    x_feat : (B, C_in, N)
    z      : (B, N)           # 좌표 (보통 [-1,1]이나 [-π,π]로 정규화)
    output : (B, C_out, N)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        *,
        separable: bool = True,
        bias: bool = True,
        temb_dim: int = 256,         
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.separable = separable

        ks = _build_modes(n_modes)
        self.register_buffer("modes", ks, persistent=False)  # (M,)
        M = ks.numel()

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "For separable=True, in_channels must equal out_channels."
                )
            # 채널별 복소 가중치 (C, M)
            self.wr = nn.Parameter(torch.randn(in_channels, M) * 0.02)
            self.wi = nn.Parameter(torch.randn(in_channels, M) * 0.02)
        else:
            # (C_out, C_in, M)
            self.wr = nn.Parameter(torch.randn(out_channels, in_channels, M) * 0.02)
            self.wi = nn.Parameter(torch.randn(out_channels, in_channels, M) * 0.02)

        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1)) if bias else None
        # --- (추가) 시간임베딩 선형층: FNO의 contract_dense와 동일 개념 ---
        self.tact  = nn.SiLU()
        self.tproj = nn.Linear(temb_dim, in_channels)
        # FNO와 동일한 초기화: 작은 균일분포 + bias=0
        self.tproj.weight.data = default_init()(self.tproj.weight.data.shape)
        nn.init.zeros_(self.tproj.bias)        

    def forward(self, x_feat: torch.Tensor, z: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_feat: (B, C_in, N)
        z     : (B, N)   # 좌표 (연속, 비균등)
        """
        B, C_in, N = x_feat.shape
        device = x_feat.device
        ks = self.modes.to(device)  # (M,)
        M = ks.shape[0]
        # --- (추가) 시간-조건화: 채널 방향으로 additive shift ---
        if temb is not None:
            shift = self.tproj(self.tact(temb))               # (B, C_in)
            x_feat = x_feat + shift.unsqueeze(-1)             # (B, C_in, N)        

        # --- 1. Forward NUFFT (real input → complex spectrum)
        # φ_minus = exp(-i k z)
        # exp(-iθ) = cos θ - i sin θ
        # cos,sin: (B, N, M)
        phase = z.unsqueeze(-1) * ks.view(1, 1, M)
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        # F = Σ_j x(j) * exp(-i k z_j) / N
        # batched matmul: (B, C, N) @ (B, N, M) -> (B, C, M)
        F_real = torch.bmm(x_feat, cos) / max(N, 1)
        F_imag = torch.bmm(x_feat, -sin) / max(N, 1)

        # --- 2. Apply complex weights in spectral domain
        if self.separable:
            # (B, C, M) ∘ (1, C, M)
            wr = self.wr.unsqueeze(0)  # (1, C, M)
            wi = self.wi.unsqueeze(0)  # (1, C, M)
            G_real = F_real * wr - F_imag * wi
            G_imag = F_real * wi + F_imag * wr
            # (B, C, M)
        else:
            # (C_out, C_in, M) ⊗ (B, C_in, M) -> (B, C_out, M)
            # G = W * F (complex matmul; 모드별 동일한 행렬)
            # 실수부: sum_c wr*out,c,m * F_r - wi*out,c,m * F_i
            # 허수부: sum_c wr*out,c,m * F_i + wi*out,c,m * F_r
            wr = self.wr  # (C_out, C_in, M)
            wi = self.wi  # (C_out, C_in, M)

            # 확장을 맞춰 연산
            Fr = F_real.unsqueeze(1)  # (B, 1, C_in, M)
            Fi = F_imag.unsqueeze(1)  # (B, 1, C_in, M)
            wr = wr.unsqueeze(0)      # (1, C_out, C_in, M)
            wi = wi.unsqueeze(0)      # (1, C_out, C_in, M)

            G_real = (Fr * wr - Fi * wi).sum(dim=2)  # (B, C_out, M)
            G_imag = (Fr * wi + Fi * wr).sum(dim=2)  # (B, C_out, M)

        # --- 3. Inverse NUFFT (complex spectrum → real signal)
        # φ_plus = exp(+i k z) = cos θ + i sin θ
        # y = Re(Σ_k G(k) * exp(i k z)) / M
        # (B, C_out, M) @ (B, M, N) -> (B, C_out, N)
        y_real = torch.bmm(G_real, cos.transpose(1, 2))  # (B, C, N)
        y_imag = torch.bmm(G_imag, sin.transpose(1, 2))  # (B, C, N)
        y = (y_real - y_imag) / max(M, 1)

        if self.bias is not None:
            y = y + self.bias
        return y


class NUFNO(nn.Module):
    """
    Non-uniform Fourier Neural Operator (1D)

    config.model에서 다음 필드 사용:
      - n_modes: [64,]
      - hidden_channels: 256
      - in_channels: 2          # 0: signal (x_t), 1: coords (x_coord)
      - out_channels: 1
      - lifting_channels: 256
      - projection_channels: 256
      - n_layers: 4
      - norm: 'group_norm'
      - preactivation: True
      - skip: 'soft-gating'
      - separable: True         # 권장 (메모리/속도)
      - use_mlp: True (optional, default 내부에서 True로 설정)
    """

    def __init__(self, config):
        super().__init__()
        cfg = config.model

        # 하이퍼파라미터
        self.n_dim = 1
        self.n_modes = cfg.n_modes
        assert isinstance(self.n_modes, (list, tuple)) and len(self.n_modes) == 1, \
            "NUFNO는 1D만 지원하므로 n_modes는 [int,] 형식이어야 합니다."

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

        # 시간 임베딩 (FNO와 동일)
        self.Dense = nn.ModuleList([
            nn.Linear(self.lifting_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        ])
        for layer in self.Dense:
            nn.init.uniform_(layer.weight, a=-1e-2, b=1e-2)
            nn.init.zeros_(layer.bias)

        # Lifting / Projection (1D Conv1d)
        self.lifting = Lifting(in_channels=self.in_channels, out_channels=self.hidden_channels, n_dim=1)
        self.projection = Projection(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_dim=1,
            non_linearity=F.gelu,
        )

        # NU-스펙트럴 블록
        self.spectral_blocks = nn.ModuleList([
            NUSpectralConv1D(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_modes=self.n_modes[0],
                separable=self.separable,
                bias=True,
                temb_dim=self.lifting_channels, 
            ) for _ in range(self.n_layers)
        ])

        # Norm & Skip & (선택) MLP
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
        # ----- learnable frequency scale (s) -----
        self.s_min = getattr(cfg, "freq_scale_min", 0.05)
        self.s_max = getattr(cfg, "freq_scale_max", 1.5)
        s_init     = getattr(cfg, "freq_scale_init", 0.3)

        self.learnable_freq_scale = getattr(cfg, "learnable_freq_scale", True)
        self.per_layer_freq_scale = getattr(cfg, "per_layer_freq_scale", False)

        def _to_logit(val: float, a: float, b: float) -> torch.Tensor:
            eps = 1e-6
            p = (val - a) / (b - a)
            p = float(min(max(p, eps), 1.0 - eps))
            import math as _m
            return torch.tensor(_m.log(p / (1.0 - p)), dtype=torch.float32)

        if self.learnable_freq_scale:
            if self.per_layer_freq_scale:
                # 레이어별 s_ell
                self._z_logit = nn.ParameterList([
                    nn.Parameter(_to_logit(s_init, self.s_min, self.s_max)) for _ in range(self.n_layers)
                ])
            else:
                # 전역 s
                self._z_logit = nn.Parameter(_to_logit(s_init, self.s_min, self.s_max))
        else:
            # 고정 하이퍼파라미터 (state_dict에는 저장되지만 학습은 안 함)
            if self.per_layer_freq_scale:
                self.register_buffer("freq_scale", torch.full((self.n_layers,), float(s_init)))
            else:
                self.register_buffer("freq_scale", torch.tensor(float(s_init)))
        # 시간 임베딩 (FNO와 동일)
        self.Dense = nn.ModuleList([
            nn.Linear(self.lifting_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        ])
        for layer in self.Dense:
            nn.init.uniform_(layer.weight, a=-1e-2, b=1e-2)
            nn.init.zeros_(layer.bias)

        # Lifting / Projection (1D Conv1d)
        self.lifting = Lifting(in_channels=self.in_channels, out_channels=self.hidden_channels, n_dim=1)
        self.projection = Projection(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_dim=1,
            non_linearity=F.gelu,
        )

        # NU-스펙트럴 블록
        self.spectral_blocks = nn.ModuleList([
            NUSpectralConv1D(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_modes=self.n_modes[0],
                separable=self.separable,
                bias=True,
                temb_dim=self.lifting_channels,
            ) for _ in range(self.n_layers)
        ])

        # Norm & Skip & (선택) MLP
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, N)
           - 마지막 채널이 좌표(x_coord)라고 가정 (입력은 [x_t, x_coord_norm] 형태)
        t: (B,)
        return: (B, N)
        """
        B, Cin, N = x.shape
        device = x.device

        # 좌표 추출 (마지막 채널, [-1,1] 정규화 가정)
        x_coord = x[:, -1, :]  # (B, N)

        # Lifting
        h = self.lifting(x)  # (B, C_hid, N)

        # 시간 임베딩
        temb = get_timestep_embedding(t, self.lifting_channels)  # (B, C_lift)
        temb = self.Dense[0](temb)
        temb = self.Dense[1](F.silu(temb))
        h = h + temb.unsqueeze(-1)  # (B, C_hid, N)

        # s 준비
        def _s_from_logit(z_logit: torch.Tensor) -> torch.Tensor:
            return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(z_logit)

        if self.learnable_freq_scale:
            if self.per_layer_freq_scale:
                s_list = [_s_from_logit(self._z_logit[i]) for i in range(self.n_layers)]  # 텐서 스칼라들
            else:
                s_global = _s_from_logit(self._z_logit)  # 텐서 스칼라
        else:
            if self.per_layer_freq_scale:
                s_list = [self.freq_scale[i] for i in range(self.n_layers)]
            else:
                s_global = self.freq_scale

        # 스택
        for i in range(self.n_layers):
            if self.preactivation:
                h = F.silu(h)
                if self.norms is not None:
                    h = self.norms[i](h)

            # NU-Spectral Block에 들어갈 위상 좌표: z = (s * π) * x_coord_norm
            if self.per_layer_freq_scale:
                s_i = s_list[i]
                z = x_coord * (s_i * math.pi)
            else:
                z = x_coord * (s_global * math.pi)

            h_f = self.spectral_blocks[i](h, z, temb)   # (B, C_hid, N)

            if not self.preactivation and self.norms is not None:
                h_f = self.norms[i](h_f)

            h = h_f + self.skips[i](h)

            if self.use_mlp:
                h = self.mlps[i](h, temb)

        y = self.projection(h).squeeze(1)  # (B, N)
        return y
