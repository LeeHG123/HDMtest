# models/nufno.py
# Non-uniform Fourier Neural Operator (NUFNO, 1D)
# - 학습 가능한 비균등 주파수 집합 {kappa_m}
# - 추가: |kappa| 기준 저/중/고 밴드 분할 + 밴드별 게이트(∈[0,1]) 적용
# - 역변환 직전, G_real/G_imag에 밴드 마스크를 곱함

import math
from typing import Optional, List, Tuple

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

# --- New: 시간 t-인식 주파수 게이트 ------------------------------------------
class ProgressiveFreqGate(nn.Module):
    """
    g_k(t) = sigmoid( alpha * ( tau(t) - |kappa_k| ) )
    tau(t) = tau0 + softplus(tau1_raw) * s(t),   s(t) = tanh( MLP(temb) )

    - alpha: 양수 기울기(softplus로 양수 보장)
    - tau0:  초기 컷오프(작게 초기화 → 저주파만 통과)
    - tau1:  학습으로 확장 폭을 자동 조절
    - MLP(temb): 시간 임베딩 → 스칼라 s(t) ∈ (-1,1)

    출력 g: (B, M) ∈ (0,1)
    """

    def __init__(
        self,
        M: int,
        temb_dim: int,
        kappa_init: torch.Tensor,
        *,
        hidden: int = 128,
        tau0_frac: float = 0.15,     # |kappa|의 q-분위수로 tau0 초기화(창을 좁게 시작)
        alpha_init: float = 8.0      # 초깃값이 너무 작으면 개방이 느려짐
    ):
        super().__init__()
        self.M = int(M)

        h = min(hidden, temb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(temb_dim, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        # FNO 초기화 방식과 일관
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                m.weight.data = default_init()(m.weight.data.shape)
                nn.init.zeros_(m.bias)

        # tau0: |kappa| q-분위수로 초기화 → 초반엔 저주파만 통과
        with torch.no_grad():
            k_abs = kappa_init.detach().abs().cpu()
            try:
                q = torch.quantile(k_abs, tau0_frac).item()
            except Exception:
                k_sorted, _ = torch.sort(k_abs)
                idx = int((len(k_sorted) - 1) * tau0_frac)
                q = k_sorted[idx].item()
        self.tau0 = nn.Parameter(torch.tensor(float(q)))

        # tau1, alpha: softplus로 양수화
        def _inv_softplus_scalar(y: float, eps: float = 1e-6) -> float:
            y = max(y, eps)
            return float(math.log(math.expm1(y)))

        self.tau1_raw  = nn.Parameter(torch.tensor(_inv_softplus_scalar(0.5)))  # softplus≈0.5
        self.alpha_raw = nn.Parameter(torch.tensor(_inv_softplus_scalar(alpha_init)))

    def forward(self, temb: torch.Tensor, kappa_curr: torch.Tensor) -> torch.Tensor:
        """
        temb:  (B, C_temb)
        kappa_curr: (M,)
        return: g (B, M) ∈ (0,1)
        """
        B = temb.size(0)
        # s(t) ∈ (-1,1)
        s = torch.tanh(self.mlp(temb)).view(B)                       # (B,)
        tau = self.tau0 + F.softplus(self.tau1_raw) * s               # (B,)
        alpha = F.softplus(self.alpha_raw) + 1e-12

        k_abs = kappa_curr.detach().abs().view(1, -1)                 # (1, M) — 안정 위해 detach
        arg = alpha * (tau.view(B, 1) - k_abs)                        # (B, M)
        return torch.sigmoid(arg)                                     # (B, M)

    # 편의: 현재 하이퍼파라미터 노출(로깅/디버깅용)
    def state(self) -> dict:
        return {
            "tau0": float(self.tau0.detach()),
            "tau1": float(F.softplus(self.tau1_raw.detach())),
            "alpha": float(F.softplus(self.alpha_raw.detach())),
        }
# ---------------------------------------------------------------------------
def _build_modes(n_modes: int) -> torch.Tensor:
    """정수 격자 모드 [-K..K] (길이 M=2K+1)."""
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")
    K = n_modes // 2
    ks = torch.arange(-K, K + 1, dtype=torch.float32)
    return ks


def _inv_softplus(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """softplus(x)=log(1+exp(x)) 역함수: x=log(exp(y)-1)."""
    y = y.clamp_min(eps)
    return torch.log(torch.expm1(y))


class NUSpectralConv1D(nn.Module):
    """
    1D Non-uniform Spectral Convolution (NUFFT 근사)
      x_feat: (B, C_in, N)
      z     : (B, N)  # 보통 z = pi * x_norm
      out   : (B, C_out, N)

    추가: 주파수 밴드 게이팅
      - |κ|의 분위수로 저/중/고 밴드를 나눠 각 밴드에 게이트 g∈[0,1]을 곱함
      - 밴드 경계는 기본 (0.4, 0.8) 분위수. 필요시 고정 경계를 사용할 수도 있음.
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
        kappa_warm_start_scale: float = 0.3,  # freq_scale_init 역할(초기 κ 스케일)
        # ---- 밴드 게이팅 옵션 ----
        enable_band_gating: bool = True,
        band_split_fracs: Tuple[float, float] = (0.4, 0.8),  # 분위수 경계 (저<=q1, 중<=q2, 고>q2)
        band_fixed_edges: Optional[Tuple[float, float]] = None,  # |κ| 고정 경계 (우선순위 ↑)
        # ---- 시간 t-인식 주파수 스케줄 ----
        enable_time_freq_gate: bool = True,
        time_gate_tau0_frac: float = 0.15,
        time_gate_alpha_init: float = 8.0,
        time_gate_hidden: int = 128,
        # ---- 가보르 창 옵션 ----
        enable_gabor_window: bool = True,
        gabor_omega_init: float = 0.25,
        gabor_centers_init: Optional[torch.Tensor] = None,  # (M,) 제공 시 사용
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.separable = separable

        # 기준 정수 격자 (정규화/정규항용)
        ks_int = _build_modes(n_modes)
        self.register_buffer("baseline_modes", ks_int, persistent=False)  # (M,)
        self.M = ks_int.numel()
        self.K = (self.M - 1) // 2  # 양의 쪽 모드 개수    
        # 학습 가능한 양의 쪽 κ: κ_pos = cumsum(softplus(a_raw))  (단조/양수)
        with torch.no_grad():
            target_pos = torch.arange(1, self.K + 1, dtype=torch.float32) * float(
                kappa_warm_start_scale
            )
            deltas = target_pos - torch.cat([torch.zeros(1), target_pos[:-1]])
            a0 = _inv_softplus(deltas)
        self.kappa_pos_raw = nn.Parameter(a0)  # (K,)
        # ===== 시간 게이트 생성 =====
        self.enable_time_freq_gate = bool(enable_time_freq_gate)
        if self.enable_time_freq_gate:
            # 현재 초기 κ로 tau0 초기화
            with torch.no_grad():
                kpos0 = torch.cumsum(F.softplus(self.kappa_pos_raw.detach()), dim=0)
                kappa_init = torch.cat([-kpos0.flip(0), torch.zeros(1), kpos0], dim=0)  # (M,)
            self.time_gate = ProgressiveFreqGate(
                M=self.M,
                temb_dim=temb_dim,
                kappa_init=kappa_init,
                hidden=time_gate_hidden,
                tau0_frac=float(time_gate_tau0_frac),
                alpha_init=float(time_gate_alpha_init),
            )
        else:
            self.time_gate = None    

        # ===== 가보르 창 파라미터 =====
        self.enable_gabor_window = bool(enable_gabor_window)
        if self.enable_gabor_window:
            # 중심 c_m 초기화: 제공 없으면 [-1,1] 선형 등분
            if gabor_centers_init is None:
                c0 = torch.linspace(-1.0, 1.0, steps=self.M, dtype=torch.float32)
            else:
                c0 = gabor_centers_init.to(torch.float32).view(-1)
                assert c0.numel() == self.M, "gabor_centers_init must have length M"
            # 폭 ω_m > 0 : softplus 파라미터화
            def _inv_softplus_scalar(y: float, eps: float = 1e-6) -> float:
                y = max(float(y), eps)
                return float(math.log(math.expm1(y)))
            omega0 = torch.full((self.M,), float(gabor_omega_init), dtype=torch.float32)
            self.gabor_c = nn.Parameter(c0)  # (M,)
            self.gabor_omega_raw = nn.Parameter(
                torch.tensor([_inv_softplus_scalar(v) for v in omega0.tolist()],
                             dtype=torch.float32)
            )
            self.gabor_eps = 1e-6  # 양수 보장용 작은 값                    

        # 복소 가중치
        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "separable=True에서는 in_channels == out_channels 이어야 합니다."
                )
            self.wr = nn.Parameter(torch.randn(in_channels, self.M) * 0.02)
            self.wi = nn.Parameter(torch.randn(in_channels, self.M) * 0.02)
        else:
            self.wr = nn.Parameter(torch.randn(out_channels, in_channels, self.M) * 0.02)
            self.wi = nn.Parameter(torch.randn(out_channels, in_channels, self.M) * 0.02)

        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1)) if bias else None

        # 시간 임베딩 → 채널 shift
        self.tact = nn.SiLU()
        self.tproj = nn.Linear(temb_dim, in_channels)
        self.tproj.weight.data = default_init()(self.tproj.weight.data.shape)
        nn.init.zeros_(self.tproj.bias)

        # ===== 밴드 게이팅 파트 =====
        self.enable_band_gating = bool(enable_band_gating)
        self.band_split_fracs = tuple(float(x) for x in band_split_fracs)
        if len(self.band_split_fracs) != 2 or not (0.0 < self.band_split_fracs[0] < self.band_split_fracs[1] < 1.0):
            raise ValueError("band_split_fracs must be two increasing fractions in (0,1), e.g., (0.4, 0.8).")
        # 고정 경계를 쓰고 싶다면 |κ| 값의 두 경계(th1, th2)를 지정 (None이면 분위수 사용)
        self.band_fixed_edges: Optional[Tuple[float, float]] = band_fixed_edges

        # 게이트는 [0,1]로 제한 (sigmoid). 초기값 g_low≈1, g_mid≈0.1, g_high≈0.1
        self.gate_low_raw  = nn.Parameter(torch.tensor(4.0))   # → sigmoid≈0.982
        self.gate_mid_raw  = nn.Parameter(torch.tensor(-2.1972246))  # → sigmoid≈0.1
        self.gate_high_raw = nn.Parameter(torch.tensor(-2.1972246))  # → sigmoid≈0.1

    # 가보르 창의 배치 계산 헬퍼 추가 
    def _compute_gabor_window(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N) with z = pi * x_norm
        return s: (B, N, M) where s[b,n,m] = exp( - (x_tilde - c_m)^2 / (2 omega_m^2) )
        x_tilde = z / pi ∈ [-1,1] (기존 정규화와 일치)
        """
        if not self.enable_gabor_window:
            # 브로드캐스트를 위해 ones 반환
            B, N = z.shape
            return torch.ones(B, N, self.M, device=z.device, dtype=z.dtype)

        x_tilde = (z / math.pi).unsqueeze(-1)              # (B,N,1)
        c = self.gabor_c.to(z.device, z.dtype).view(1, 1, self.M)  # (1,1,M)
        omega = F.softplus(self.gabor_omega_raw.to(z.device, z.dtype)) \
                .clamp_min(self.gabor_eps).view(1, 1, self.M)       # (1,1,M)
        s = torch.exp(-0.5 * ((x_tilde - c) / omega) ** 2)          # (B,N,M)
        return s

    # ----- κ 구성/공개 -----
    def _kappa_pos(self) -> torch.Tensor:
        """양의 쪽 κ (K,) = cumsum(softplus(a_raw))."""
        return torch.cumsum(F.softplus(self.kappa_pos_raw), dim=0)

    def kappa_full(self) -> torch.Tensor:
        """전체 κ (M,) = [-k_pos[::-1], 0, k_pos]."""
        kpos = self._kappa_pos()
        return torch.cat([-kpos.flip(0), torch.zeros(1, device=kpos.device), kpos], dim=0)

    def kappa_for_reg(self) -> torch.Tensor:
        """정규화항 계산용 현재 κ (M,)."""
        return self.kappa_full()

    # ----- 게이트 유틸 -----
    def gates(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(g_low, g_mid, g_high) 각 ∈ (0,1)."""
        return (
            torch.sigmoid(self.gate_low_raw),
            torch.sigmoid(self.gate_mid_raw),
            torch.sigmoid(self.gate_high_raw),
        )

    def gate_regularizer(self) -> torch.Tensor:
        """
        선택적 정규화 항: g_low가 1 근방에 머물도록 작은 ℓ² 페널티.
        (외부 학습 루프에서 loss에 더해 사용. 기본적으로 호출되지 않음.)
        """
        g_low, _, _ = self.gates()
        return (g_low - 1.0).pow(2)

    @staticmethod
    def _quantile_detached(x: torch.Tensor, q: float) -> torch.Tensor:
        """torch.quantile 사용(가능 시) + κ 경계는 gradient 차단."""
        x_ = x.detach()
        try:
            return torch.quantile(x_, q)
        except Exception:
            # PyTorch 구버전 대비: 정렬 후 위치 보간
            k = int((len(x_) - 1) * q)
            x_sorted, _ = torch.sort(x_)
            return x_sorted[k]

    def _compute_band_mask(self, ks: torch.Tensor) -> torch.Tensor:
        """
        입력: ks (M,)  — 현재 κ 배열
        반환: mask (M,) — 각 모드에 곱할 게이트 값
        """
        if not self.enable_band_gating:
            return torch.ones_like(ks)

        k_abs = ks.detach().abs()  # 밴드 경계는 학습 안정 위해 κ 그래디언트 차단

        if self.band_fixed_edges is not None:
            th1, th2 = float(self.band_fixed_edges[0]), float(self.band_fixed_edges[1])
            th1 = torch.as_tensor(th1, device=k_abs.device, dtype=k_abs.dtype)
            th2 = torch.as_tensor(th2, device=k_abs.device, dtype=k_abs.dtype)
        else:
            q1, q2 = self.band_split_fracs
            th1 = self._quantile_detached(k_abs, q1)
            th2 = self._quantile_detached(k_abs, q2)

        g_low, g_mid, g_high = self.gates()  # 스칼라 텐서들

        # 밴드별 마스크 작성
        mask = torch.empty_like(k_abs)
        low  = (k_abs <= th1)
        mid  = (k_abs > th1) & (k_abs <= th2)
        high = (k_abs > th2)
        mask[low]  = g_low
        mask[mid]  = g_mid
        mask[high] = g_high
        return mask
    
    # forward ─────────────────────────────────────
    def forward(self, x_feat: torch.Tensor, z: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C_in, N = x_feat.shape
        device = x_feat.device

        # 시간조건: additive shift 
        if temb is not None:
            shift = self.tproj(self.tact(temb))  # (B, C_in)
            x_feat = x_feat + shift.unsqueeze(-1)

        # 주파수/위상 준비 
        ks = self.kappa_full().to(device)  # (M,)
        phase = z.unsqueeze(-1) * ks.view(1, 1, self.M)  # (B, N, M)
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        # ===== 가보르 창 계산 및 적용 =====
        s_win = self._compute_gabor_window(z)   # (B, N, M)
        cos_w = cos * s_win
        sin_w = sin * s_win

        # Forward NUDFT with window
        # x_feat: (B, C_in, N)  · (B, N, M) → (B, C_in, M)
        Fr = torch.bmm(x_feat,      cos_w).div(max(N, 1))
        Fi = torch.bmm(x_feat, -1 * sin_w).div(max(N, 1))

        # 복소 가중치
        if self.separable:
            wr = self.wr.unsqueeze(0)  # (1, C, M)
            wi = self.wi.unsqueeze(0)  # (1, C, M)
            Gr = Fr * wr - Fi * wi                     # (B, C, M)
            Gi = Fr * wi + Fi * wr                     # (B, C, M)
        else:
            Fr_e = Fr.unsqueeze(1)  # (B, 1, C_in, M)
            Fi_e = Fi.unsqueeze(1)  # (B, 1, C_in, M)
            wr = self.wr.unsqueeze(0)  # (1, C_out, C_in, M)
            wi = self.wi.unsqueeze(0)  # (1, C_out, C_in, M)
            Gr = (Fr_e * wr - Fi_e * wi).sum(dim=2)    # (B, C_out, M)
            Gi = (Fr_e * wi + Fi_e * wr).sum(dim=2)    # (B, C_out, M)

        # 밴드 게이트 / 시간 게이트 적용
        if self.enable_band_gating:
            band_mask = self._compute_band_mask(ks).view(1, 1, self.M)
            Gr = Gr * band_mask
            Gi = Gi * band_mask
        if self.enable_time_freq_gate and (temb is not None):
            g_time = self.time_gate(temb, ks).view(B, 1, self.M)
            Gr = Gr * g_time
            Gi = Gi * g_time

        # Inverse NUDFT with window (창을 역변환에도 곱해 합성)
        # (B, C_out, M) · (B, M, N) → (B, C_out, N)
        yr = torch.bmm(Gr, cos_w.transpose(1, 2))
        yi = torch.bmm(Gi, sin_w.transpose(1, 2))
        y = (yr - yi).div(max(self.M, 1))  # 정규화는 기존 규칙 유지

        if self.bias is not None:
            y = y + self.bias
        return y


class NUFNO(nn.Module):
    """
    1D Non-uniform FNO
      입력 채널 2개 [signal, coord_norm]
      시간임베딩, GroupNorm, soft-gating skip, (선택) MLP 유지
      각 레이어가 자기 κ 집합과 (추가) 밴드 게이트를 학습
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

        # 시간 임베딩
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

        # NU-스펙트럴 블록 (레이어별 κ + 밴드 게이트)
        kappa_warm = float(getattr(cfg, "freq_scale_init", 0.3))
        # 밴드 분할/게이팅 기본값은 논문 캡션의 계획을 따름 (0~40%~80%~100%)
        band_fracs = getattr(cfg, "band_split_fracs", (0.4, 0.8))
        use_gating = bool(getattr(cfg, "enable_band_gating", True))
        # --- 신규: 시간 게이트 하이퍼(옵션)
        use_time_gate = bool(getattr(cfg, "enable_time_freq_gate", True))
        tg_tau0_frac  = float(getattr(cfg, "time_gate_tau0_frac", 0.15))
        tg_alpha_init = float(getattr(cfg, "time_gate_alpha_init", 8.0))
        tg_hidden     = int(getattr(cfg, "time_gate_hidden", 128))        
        use_gabor = bool(getattr(cfg, "enable_gabor_window", True))
        gabor_omega_init = float(getattr(cfg, "gabor_omega_init", 0.25))        
        self.spectral_blocks = nn.ModuleList([
            NUSpectralConv1D(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_modes=self.n_modes[0],
                separable=self.separable,
                bias=True,
                temb_dim=self.lifting_channels,
                kappa_warm_start_scale=kappa_warm,
                enable_band_gating=use_gating,
                band_split_fracs=band_fracs,
                band_fixed_edges=None,   # 필요시 (th1, th2)로 고정 가능
                # ---- 시간 게이트 전달 ----
                enable_time_freq_gate=use_time_gate,
                time_gate_tau0_frac=tg_tau0_frac,
                time_gate_alpha_init=tg_alpha_init,
                time_gate_hidden=tg_hidden,        
                # ---- 가보르 창 전달 ----
                enable_gabor_window=use_gabor,
                gabor_omega_init=gabor_omega_init,        
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

    # 정규화/로깅용
    def all_kappas(self) -> List[torch.Tensor]:
        """모든 레이어의 현재 κ (M,) 리스트."""
        return [blk.kappa_for_reg() for blk in self.spectral_blocks]

    def all_baselines(self) -> List[torch.Tensor]:
        """모든 레이어의 baseline 정수 모드 (M,) 리스트."""
        return [blk.baseline_modes for blk in self.spectral_blocks]

    def all_band_gates(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """각 레이어 (g_low, g_mid, g_high) 목록(모두 ∈(0,1))."""
        return [blk.gates() for blk in self.spectral_blocks]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, N)  — C_in=2: [signal, coord_norm]
        t: (B,)
        return: (B, N)
        """
        B, Cin, N = x.shape

        # 좌표 채널 (러너가 [-1,1]로 정규화한 값)
        x_coord_norm = x[:, -1, :]  # (B, N)

        # Lifting
        h = self.lifting(x)  # (B, C_hid, N)

        # 시간 임베딩
        temb = get_timestep_embedding(t, self.lifting_channels)  # (B, C_lift)
        temb = self.Dense[0](temb)
        temb = self.Dense[1](F.silu(temb))
        h = h + temb.unsqueeze(-1)  # (B, C_hid, N)

        # 위상용 좌표 z = pi * x_norm  (전역 스케일 s 제거)
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



