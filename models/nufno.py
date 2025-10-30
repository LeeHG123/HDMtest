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
        weighted_normalization: bool = True,
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
        measure_scale: float = 1.0,
        use_symmetric_trapz_sqrt: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.separable = separable
        self.measure_scale = float(measure_scale)
        self.weighted_normalization = bool(weighted_normalization)
        self.use_symmetric_trapz_sqrt = bool(use_symmetric_trapz_sqrt)
        self._w_eps = 1e-12        

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
        import math
        B, C_in, N = x_feat.shape
        device = x_feat.device

        # (1) temb 채널 shift
        if temb is not None:
            shift = self.tproj(self.tact(temb))        # (B, C_in)
            x_feat = x_feat + shift.unsqueeze(-1)      # (B, C_in, N)

        # (2) 위상/기저
        ks   = self.kappa_full().to(device)                           # (M,)
        phase = z.unsqueeze(-1) * ks.view(1, 1, self.M)               # (B, N, M)
        cos  = torch.cos(phase)
        sin  = torch.sin(phase)

        # (3) 가보르 창
        s_win = self._compute_gabor_window(z)                         # (B, N, M)
        cos_w = cos * s_win
        sin_w = sin * s_win

        # (4) 사다리꼴 가중 및 정규화
        x_norm = (z / math.pi)                                        # (B, N) in [-1,1]
        w_x    = _trapz_weights_1d(x_norm) * self.measure_scale       # (B, N)
        if self.weighted_normalization:
            L = w_x.sum(dim=1, keepdim=True).clamp_min(1e-12)         # (B, 1)
            sym_frfi = L.rsqrt().view(B, 1, 1)                        # (B,1,1)
            sym_y    = L.rsqrt().view(B, 1, 1)                        # (B,1,1)
        else:
            sym = (float(N)) ** (-0.5)
            sym_frfi = x_feat.new_full((B,1,1), sym)                  # (B,1,1)
            sym_y    = x_feat.new_full((B,1,1), sym)                  # (B,1,1)

        # (5) ★ √(trapz) 대칭화 적용
        if self.use_symmetric_trapz_sqrt:
            # √w (수치안정)
            sqrt_w = torch.sqrt(w_x.clamp_min(self._w_eps))           # (B, N)

            # 전방: 입력과 기저 양쪽에 √w
            x_feat_w = x_feat * sqrt_w.unsqueeze(1)                   # (B, C_in, N)
            cos_sw   = cos_w * sqrt_w.unsqueeze(-1)                   # (B, N, M)
            sin_sw   = sin_w * sqrt_w.unsqueeze(-1)                   # (B, N, M)

            # NUDFT(가중, 정규화 포함)
            Fr = torch.bmm(x_feat_w,      cos_sw) * sym_frfi          # (B, C_in, M)
            Fi = torch.bmm(x_feat_w, -1 * sin_sw) * sym_frfi          # (B, C_in, M)

        else:
            # 기존(= w를 전방 합에 직접 곱하는 방식)
            w_x_fwd = w_x.unsqueeze(-1)                               # (B, N, 1)
            Fr = torch.bmm(x_feat,      cos_w * w_x_fwd) * sym_frfi   # (B, C_in, M)
            Fi = torch.bmm(x_feat, -1 * sin_w * w_x_fwd) * sym_frfi   # (B, C_in, M)

        # (6) 복소 가중 채널 혼합
        if self.separable:
            wr = self.wr.unsqueeze(0)                                 # (1, C_in==C_out, M)
            wi = self.wi.unsqueeze(0)
            Gr = Fr * wr - Fi * wi                                    # (B, C_out, M)
            Gi = Fr * wi + Fi * wr
        else:
            Fr_e = Fr.unsqueeze(1)                                    # (B, 1, C_in, M)
            Fi_e = Fi.unsqueeze(1)                                    # (B, 1, C_in, M)
            wr   = self.wr.unsqueeze(0)                               # (1, C_out, C_in, M)
            wi   = self.wi.unsqueeze(0)
            Gr = (Fr_e * wr - Fi_e * wi).sum(dim=2)                   # (B, C_out, M)
            Gi = (Fr_e * wi + Fi_e * wr).sum(dim=2)

        # (7) 역변환(공액전치) + 정규화
        if self.use_symmetric_trapz_sqrt:
            # ★ 전치 기저에도 √w를 대칭 적용
            cos_sw_T = cos_sw.transpose(1, 2)                         # (B, M, N)
            sin_sw_T = sin_sw.transpose(1, 2)                         # (B, M, N)
            yr = torch.bmm(Gr, cos_sw_T)                              # (B, C_out, N)
            yi = torch.bmm(Gi, sin_sw_T)                              # (B, C_out, N)
            y  = (yr - yi) * sym_y                                    # (B, C_out, N)
        else:
            cos_w_T = cos_w.transpose(1, 2)                           # (B, M, N)
            sin_w_T = sin_w.transpose(1, 2)                           # (B, M, N)
            yr = torch.bmm(Gr, cos_w_T)                               # (B, C_out, N)
            yi = torch.bmm(Gi, sin_w_T)                               # (B, C_out, N)
            y_tmp = (yr - yi) * sym_y                                 # (B, C_out, N)
            y     = y_tmp * w_x.unsqueeze(1)                          # (B, C_out, N)  ← 기존 경로

        if self.bias is not None:
            y = y + self.bias                                         # (B, C_out, N)

        return y

# 1D trapezoid weights on already-sorted grid
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

class _Pos(nn.Module):
    def forward(self, x):              # stable softplus -> positive
        return F.softplus(x, beta=1.0) + 1e-8

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                # light, stable init
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

class _FeatureNet1D(nn.Module):
    # x -> local parameters for NS-GSM
    def __init__(self, hidden=64, Q=2):
        super().__init__()
        self.Q = Q
        self.core = _MLP(in_dim=1, hidden=hidden, out_dim=3*Q)  # (w,sigma,mu) per component
        self.pos = _Pos()
    def forward(self, x_bn1):  # (B,N,1)
        out = self.core(x_bn1) # (B,N,3Q)
        Q = self.Q
        w, sig, mu = out[..., :Q], out[..., Q:2*Q], out[..., 2*Q:3*Q]
        return self.pos(w), self.pos(sig), self.pos(mu)

class KNOScalarKernel1D(nn.Module):
    """
    Learnable 1D kernels:
      - RBF : k(x,y) = gain * exp(-0.5*(x-y)^2 / ℓ^2)
      - GSM : sum_q w_q * exp(-0.5 * (σ_q (x-y))^2) * cos(2π μ_q (x-y))
      - NS-GSM : x-의존 w_q(x), σ_q(x), μ_q(x) (Gibbs form)
    Optionally time-conditioned by temb (global, small hyper-MLP outputs deltas in log-parameter space).
    """
    def __init__(self,
                 kernel_type: str = "gsm",
                 Q: int = 2,
                 x_hidden: int = 64,
                 use_time_cond: bool = True,
                 temb_dim: int = 256,
                 t_hidden: int = 64):
        super().__init__()
        self.kind = kernel_type.lower()
        self.Q = Q
        self.pos = _Pos()

        # base static parameters
        if self.kind == "rbf":
            self.log_gain = nn.Parameter(torch.tensor(0.0))
            self.log_len  = nn.Parameter(torch.tensor(math.log(0.25)))
        elif self.kind == "gsm":
            self.log_w   = nn.Parameter(torch.zeros(Q))
            self.log_sig = nn.Parameter(torch.zeros(Q))
            self.log_mu  = nn.Parameter(torch.zeros(Q))
        elif self.kind == "nsgsm":
            self.feat = _FeatureNet1D(hidden=x_hidden, Q=Q)
        else:
            raise ValueError(f"unknown kernel_type={kernel_type}")

        # (optional) time-conditioning
        self.use_time_cond = bool(use_time_cond)
        if self.use_time_cond:
            if self.kind == "rbf":
                # two deltas: d(log_gain), d(log_len)
                self.tmlp = _MLP(temb_dim, t_hidden, 2)
            elif self.kind == "gsm":
                # 3Q deltas
                self.tmlp = _MLP(temb_dim, t_hidden, 3*Q)
            else:  # nsgsm: global gain + local scaling gates
                self.tmlp = _MLP(temb_dim, t_hidden, 2*Q)  # gates for w,sigma (per component)
        else:
            self.tmlp = None

    def forward(self, x: torch.Tensor, y: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x, y  : (B,N) normalized coords (we use z/pi so x,y in [-1,1])
        temb  : (B, C_temb) or None
        return: (B, N, N)
        """
        B, N = x.shape
        x_ = x.unsqueeze(2)  # (B,N,1)
        y_ = y.unsqueeze(1)  # (B,1,N)

        if self.kind == "rbf":
            log_gain = self.log_gain
            log_len  = self.log_len
            if self.use_time_cond and (temb is not None):
                delta = self.tmlp(temb)                       # (B,2)
                # limit modulation magnitude for stability
                delta = 1.5 * torch.tanh(delta)
                d_gain, d_len = delta[:, 0], delta[:, 1]      # (B,)
                gain = self.pos(log_gain + d_gain.view(B,1))  # (B,1)
                ell  = self.pos(log_len  + d_len.view(B,1))   # (B,1)
            else:
                gain = self.pos(log_gain).view(1,1)
                ell  = self.pos(log_len).view(1,1)

            diff2 = (x_ - y_)**2
            K = gain.view(B if gain.shape[0]==B else 1, 1, 1) * torch.exp(-0.5 * diff2 / (ell.view(B if ell.shape[0]==B else 1,1,1)**2))
            if K.shape[0] == 1:  # broadcast for B>1
                K = K.expand(B, N, N)
            return K

        if self.kind == "gsm":
            log_w, log_sig, log_mu = self.log_w, self.log_sig, self.log_mu
            if self.use_time_cond and (temb is not None):
                delta = self.tmlp(temb)                      # (B,3Q)
                delta = 1.5 * torch.tanh(delta)
                dw, ds, dm = torch.split(delta, [self.Q, self.Q, self.Q], dim=-1)  # (B,Q) each
                w  = self.pos(log_w.view(1,-1)   + dw)       # (B,Q)
                sig= self.pos(log_sig.view(1,-1) + ds)       # (B,Q)
                mu = self.pos(log_mu.view(1,-1)  + dm)       # (B,Q)
            else:
                w  = self.pos(log_w).view(1, self.Q).expand(B, -1)    # (B,Q)
                sig= self.pos(log_sig).view(1, self.Q).expand(B, -1)
                mu = self.pos(log_mu).view(1, self.Q).expand(B, -1)

            diff = (x_ - y_).unsqueeze(-1)       # (B,N,N,1)
            gaus = torch.exp(-0.5 * (sig.view(B,1,1,self.Q) * diff).pow(2))     # (B,N,N,Q)
            osc  = torch.cos(2*math.pi * (mu.view(B,1,1,self.Q) * diff))        # (B,N,N,Q)
            K = (w.view(B,1,1,self.Q) * gaus * osc).sum(dim=-1)                  # (B,N,N)
            return K

        # NS-GSM (nonstationary)
        wx, sx, mux = self.feat(x.unsqueeze(-1))  # (B,N,Q) each
        wy, sy, muy = self.feat(y.unsqueeze(-1))  # (B,N,Q)

        if self.use_time_cond and (temb is not None):
            gate = self.tmlp(temb)                       # (B,2Q)
            gate = torch.tanh(gate)                      # [-1,1]
            gw, gs = gate[:, :self.Q], gate[:, self.Q:]  # (B,Q)
            wx = wx * (1 + 0.25 * gw.unsqueeze(1))       # mild modulation
            sx = sx * (1 + 0.25 * gs.unsqueeze(1))
            wy = wy * (1 + 0.25 * gw.unsqueeze(1))
            sy = sy * (1 + 0.25 * gs.unsqueeze(1))

        # broadcast
        wx = wx.unsqueeze(2)  # (B,N,1,Q)
        wy = wy.unsqueeze(1)  # (B,1,N,Q)
        sx = sx.unsqueeze(2)  # (B,N,1,Q)
        sy = sy.unsqueeze(1)  # (B,1,N,Q)
        mux= mux.unsqueeze(2) # (B,N,1,Q)
        muy= muy.unsqueeze(1) # (B,1,N,Q)

        r = sx*sx + sy*sy                                     # (B, N, N, Q)
        dx2 = (x_ - y_)**2                                    # (B, N, N)
        r_eps = r + 1e-8
        gibbs = torch.sqrt(2*sx*sy / r_eps) * torch.exp(- dx2.unsqueeze(-1) / r_eps)   # (B, N, N, Q)
        phase = torch.cos(2*math.pi * (mux * x_.unsqueeze(-1) - muy * y_.unsqueeze(-1)))  # (B, N, N, Q)
        K = (wx*wy * gibbs * phase).sum(dim=-1)                                          # (B,N,N)
        return K

# === KNO spectral-integral block: y = P1x1( ∑_j K(x_i, x_j) w_j f(x_j) ) ===
class KNOSpectralIntegral1D(nn.Module):
    def __init__(self,
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
                 measure_scale: float = 1.0):
        super().__init__()
        self.kern = KNOScalarKernel1D(kernel_type=kernel_type,
                                      Q=kernel_Q,
                                      x_hidden=kernel_x_hidden,
                                      use_time_cond=enable_kernel_time_cond,
                                      temb_dim=temb_dim,
                                      t_hidden=kernel_time_hidden)
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1)) if bias else None
        self.measure_scale = float(measure_scale)

        # time-to-channel shift for inputs (same design as original NUFNO)
        self.tact  = nn.SiLU()
        self.tproj = nn.Linear(temb_dim, in_channels)
        nn.init.xavier_uniform_(self.tproj.weight); nn.init.zeros_(self.tproj.bias)

    def forward(self, x_feat: torch.Tensor, z: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x_feat: (B, C_in, N)
        z     : (B, N)   with z = pi * x_norm
        temb  : (B, C_temb)
        """
        B, C, N = x_feat.shape
        # channel shift by temb
        if temb is not None:
            shift = self.tproj(self.tact(temb))     # (B,C)
            x_feat = x_feat + shift.unsqueeze(-1)

        x_norm = z / math.pi                        # [-1,1]
        w = _trapz_weights_1d(x_norm) * self.measure_scale               # (B,N)
        K = self.kern(x_norm, x_norm, temb)         # (B,N,N)

        # integral approximation:  y = K @ (w ⊙ f)
        xw = x_feat * w.unsqueeze(1)                # (B,C,N)
        y  = torch.einsum('bmn,bcn->bcm', K, xw)    # (B,C,N)
        y  = y + self.pw(x_feat)                            # channel mixing
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
        # ---시간 게이트 하이퍼
        use_time_gate = bool(getattr(cfg, "enable_time_freq_gate", True))
        tg_tau0_frac  = float(getattr(cfg, "time_gate_tau0_frac", 0.15))
        tg_alpha_init = float(getattr(cfg, "time_gate_alpha_init", 8.0))
        tg_hidden     = int(getattr(cfg, "time_gate_hidden", 128))        
        use_gabor = bool(getattr(cfg, "enable_gabor_window", True))
        gabor_omega_init = float(getattr(cfg, "gabor_omega_init", 0.25))        
        kno_type    = str(getattr(cfg, "kernel_type", "gsm")).lower()     # 'rbf' | 'gsm' | 'nsgsm'
        kno_Q       = int(getattr(cfg, "kernel_Q", 2))
        kno_hidden  = int(getattr(cfg, "kernel_hidden", 64))
        kno_t_hidden= int(getattr(cfg, "kernel_time_hidden", 64))
        kno_t_cond  = bool(getattr(cfg, "enable_kernel_time_cond", True))

        measure_scale = float(getattr(cfg, "measure_scale", 1.0))
        weighted_norm = bool(getattr(cfg, "enable_weighted_normalization", True))

        self.spectral_blocks = nn.ModuleList([
            NUSpectralConv1D(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_modes=self.n_modes[0],
                separable=self.separable,
                bias=True,
                temb_dim=self.lifting_channels,
                kappa_warm_start_scale=float(getattr(cfg, "freq_scale_init", 0.3)),
                weighted_normalization=weighted_norm,
                enable_band_gating=bool(getattr(cfg, "enable_band_gating", True)),
                band_split_fracs=tuple(getattr(cfg, "band_split_fracs", (0.4, 0.8))),
                band_fixed_edges=None,
                enable_time_freq_gate=bool(getattr(cfg, "enable_time_freq_gate", True)),
                time_gate_tau0_frac=float(getattr(cfg, "time_gate_tau0_frac", 0.15)),
                time_gate_alpha_init=float(getattr(cfg, "time_gate_alpha_init", 8.0)),
                time_gate_hidden=int(getattr(cfg, "time_gate_hidden", 128)),
                enable_gabor_window=bool(getattr(cfg, "enable_gabor_window", True)),
                gabor_omega_init=float(getattr(cfg, "gabor_omega_init", 0.25)),
                measure_scale=measure_scale,
            )
            for _ in range(self.n_layers)
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
    def all_kappas(self):
        ks = []
        for blk in self.spectral_blocks:
            if hasattr(blk, "kappa_for_reg"):      # NUDFT 블록만
                ks.append(blk.kappa_for_reg())
        return ks

    def all_baselines(self):
        bs = []
        for blk in self.spectral_blocks:
            if hasattr(blk, "baseline_modes"):     # NUDFT 블록만
                bs.append(blk.baseline_modes)
        return bs

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

class KNO(nn.Module):
    """
    1D Kernel-Integral Operator (KNO) FNO
      입력 채널 2개 [signal, coord_norm]
      시간임베딩, GroupNorm, soft-gating skip, (선택) MLP 유지
      각 레이어가 KNO 커널 적분 블록만 사용
    """
    def __init__(self, config):
        super().__init__()
        cfg = config.model

        self.n_dim = 1
        self.n_modes = cfg.n_modes
        assert isinstance(self.n_modes, (list, tuple)) and len(self.n_modes) == 1, \
            "KNO는 1D만 지원하므로 n_modes는 [int,] 형식이어야 합니다."
        self.hidden_channels = cfg.hidden_channels
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.lifting_channels = cfg.lifting_channels
        self.projection_channels = cfg.projection_channels
        self.n_layers = cfg.n_layers
        self.norm_type = getattr(cfg, "norm", "group_norm")
        self.preactivation = getattr(cfg, "preactivation", True)
        self.skip_type = getattr(cfg, "skip", "soft-gating")

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

        # 좌표 스케일
        measure_scale = float(getattr(cfg, "measure_scale", 1.0))

        # KNO 하이퍼
        kno_type     = str(getattr(cfg, "kernel_type", "gsm")).lower()     # 'rbf' | 'gsm' | 'nsgsm'
        kno_Q        = int(getattr(cfg, "kernel_Q", 2))
        kno_hidden   = int(getattr(cfg, "kernel_hidden", 64))
        kno_t_hidden = int(getattr(cfg, "kernel_time_hidden", 64))
        kno_t_cond   = bool(getattr(cfg, "enable_kernel_time_cond", True))

        # KNO 스펙트럴 블록
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
            )
            for _ in range(self.n_layers)
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

    # NUFNO와 인터페이스를 맞추기 위한 모니터링용 메서드(빈 리스트 반환)
    def all_kappas(self):      return []
    def all_baselines(self):   return []
    def all_band_gates(self):  return []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, Cin, N = x.shape
        x_coord_norm = x[:, -1, :]  # (B, N)

        h = self.lifting(x)  # (B, C_hid, N)

        temb = get_timestep_embedding(t, self.lifting_channels)
        temb = self.Dense[0](temb)
        temb = self.Dense[1](F.silu(temb))
        h = h + temb.unsqueeze(-1)

        z = math.pi * x_coord_norm  # (B, N)

        for i in range(self.n_layers):
            if self.preactivation:
                h = F.silu(h)
                if self.norms is not None:
                    h = self.norms[i](h)

            h_f = self.spectral_blocks[i](h, z, temb)  # KNO integral

            if not self.preactivation and self.norms is not None:
                h_f = self.norms[i](h_f)

            h = h_f + self.skips[i](h)

            if self.use_mlp:
                h = self.mlps[i](h, temb)

        y = self.projection(h).squeeze(1)  # (B, N)
        return y




