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

    def forward(self, x_feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x_feat: (B, C_in, N)
        z     : (B, N)   # 좌표 (연속, 비균등)
        """
        B, C_in, N = x_feat.shape
        device = x_feat.device
        ks = self.modes.to(device)  # (M,)
        M = ks.shape[0]

        # --- 1. Forward NUFFT (real input → complex spectrum)
        # φ_minus = exp(-i k z)
        # exp(-iθ) = cos θ - i sin θ
        # cos,sin: (B, N, M)
        phase = z.unsqueeze(-1) * ks.view(1, 1, M)
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        # ---- forward NUFFT: F(k_m) ≈ ∑_j f(z_j) e^{-ik_m z_j} w_j  ----
        # z: (B, N) sorted, k: (B, M) sorted
        with torch.no_grad():
            if N > 1:  
                dz = z[:, 1:] - z[:, :-1]                 # (B, N-1)
                wz = torch.zeros_like(z)                  # (B, N)
                wz[:, 1:-1] = 0.5 * (dz[:, 1:] + dz[:, :-1])
                wz[:, 0]    = 0.5 * dz[:, 0]
                wz[:, -1]   = 0.5 * dz[:, -1]
                wz = torch.clamp(wz, min=0.0)
            else:
                wz = torch.ones_like(z)        

        # x_feat: (B, C, N), cos/sin: (B, N, M) with cos = cos(k⊗z), sin = sin(k⊗z)
        xw = x_feat * wz.unsqueeze(1)                 # (B, C, N)
        F_real = torch.bmm(xw,  cos)                  # (B, C, M)
        F_imag = torch.bmm(xw, -sin)                  # (B, C, M)  # note the minus sign for e^{-ikz}

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

        # --- 3) Inverse NUFFT (complex spectrum → real signal)
        # y(z_i) ≈ (1/2π) ∑_m G(k_m) e^{+ik_m z_i} v_m 의 실수부
        # k-측 사다리꼴 가중치(v_m ≃ Δk_m) 및 1/(2π) 적용
        with torch.no_grad():
            k_batch = ks.view(1, M).expand(B, -1)           # (B, M)
            if M > 1:
                dk = k_batch[:, 1:] - k_batch[:, :-1]       # (B, M-1)
                vk = torch.zeros_like(k_batch)              # (B, M)
                vk[:, 1:-1] = 0.5 * (dk[:, 1:] + dk[:, :-1])
                vk[:, 0]    = 0.5 * dk[:, 0]
                vk[:, -1]   = 0.5 * dk[:, -1]
                vk = torch.clamp(vk, min=0.0)
            else:
                vk = torch.ones_like(k_batch)

        # 역합성에 v_m 곱
        Gr = G_real * vk.unsqueeze(1)                       # (B, C, M)
        Gi = G_imag * vk.unsqueeze(1)                       # (B, C, M)

        # φ_plus = exp(+ikz) = cos + i sin
        # 실수부: Gr·cos - Gi·sin
        y_real = torch.bmm(Gr, cos.transpose(1, 2))         # (B, C, N)
        y_imag = torch.bmm(Gi, sin.transpose(1, 2))         # (B, C, N)

        # 기존 /M 제거, 1/(2π) 적용 → 해상도 불변 진폭
        y = (y_real - y_imag) / (2.0 * math.pi)             # (B, C, N)

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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, N)
           - 마지막 채널이 좌표(x_coord)라고 가정 (히스토리와 동일: [x_t, x_coord])
        t: (B,)
        return: (B, N)
        """
        B, Cin, N = x.shape
        device = x.device

        # 좌표 추출 (마지막 채널)
        x_coord = x[:, -1, :]  # (B, N)

        # 좌표 정규화: [-10,10] → [-π, π]로 맵핑(러프하게)
        # 기존 코드에서 x_coord는 10으로 나눠 [-1,1]였으므로, 이 값에 π를 곱해도 됨.
        # (정밀한 스케일은 학습으로 보정됨)
        z = x_coord * math.pi # (B, N)

        # Lifting
        h = self.lifting(x)  # (B, C_hid, N)

        # 시간 임베딩
        temb = get_timestep_embedding(t, self.lifting_channels)  # (B, C_lift)
        temb = self.Dense[0](temb)
        temb = self.Dense[1](self.act(temb))
        h = h + temb.unsqueeze(-1)  # (B, C_hid, N)

        # 스택
        for i in range(self.n_layers):
            if self.preactivation:
                h = self.act(h)
                if self.norms is not None:
                    h = self.norms[i](h)

            # NU-Spectral Block (coords: z)
            h_f = self.spectral_blocks[i](h, z)  # (B, C_hid, N)

            if not self.preactivation and self.norms is not None:
                h_f = self.norms[i](h_f)

            h = h_f + self.skips[i](h)

            if self.use_mlp:
                # Pre-activation 스타일 MLP
                h = self.mlps[i](h, temb)

        y = self.projection(h).squeeze(1)  # (B, N)
        return y
