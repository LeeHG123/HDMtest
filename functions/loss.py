# functions/loss.py
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def loss_fn(model, sde, x_0, t, e):
    x_mean = sde.diffusion_coeff(x_0, t)  # (사용 안함: 레거시)
    noise = sde.marginal_std(e, t)
    x_t = x_mean + noise
    score = -noise
    output = model(x_t, t)
    loss = (output - score).square().sum(dim=(1,2,3)).mean(dim=0)
    return loss

def _trapz_weights(x: torch.Tensor) -> torch.Tensor:
    """(B,N) 좌표에 대한 사다리꼴 가중치."""
    assert x.dim() == 2, "x must be (B, N)"
    dx = (x[:, 1:] - x[:, :-1]).abs()
    w = torch.zeros_like(x)
    if x.size(1) > 2:
        w[:, 1:-1] = 0.5 * (dx[:, 1:] + dx[:, :-1])
    w[:, 0]  = 0.5 * dx[:, 0]
    w[:, -1] = 0.5 * dx[:, -1]
    return w

# ─────────────────────────────────────────────────────────────
# NUDFT 유틸 (loss에서 "Forward"만 1회 더 호출)
# ─────────────────────────────────────────────────────────────
def _iter_kappas_with_meta(model):
    """
    모델 안의 모든 스펙트럴 블록을 순회하며 (kappa, split_fracs)를 반환.
    - κ는 손실에서 역전파하지 않도록 detach()합니다.
    - 스펙트럴 블록이 없으면 빈 리스트를 반환합니다.
    """
    mm = model.module if hasattr(model, "module") else model
    res = []
    if hasattr(mm, "spectral_blocks") and len(mm.spectral_blocks) > 0:
        for blk in mm.spectral_blocks:
            if hasattr(blk, "kappa_full"):
                kappa = blk.kappa_full().detach()
                split_fracs = getattr(blk, "band_split_fracs", (0.4, 0.8))
                res.append((kappa, split_fracs))
    return res

def _forward_nudft_real(r: torch.Tensor, z: torch.Tensor, kappa: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    r : (B,N) 실수 잔차
    z : (B,N) 위상 좌표 (여기서는 z = π * x_norm)
    kappa : (M,) 비균등 주파수
    반환: (Fr, Fi) 각 (B,M) — NUDFT(실수/허수)
    """
    B, N = r.shape
    M = int(kappa.numel())
    phase = z.unsqueeze(-1) * kappa.view(1, 1, M)      # (B,N,M)
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    x = r.unsqueeze(1)                                 # (B,1,N)
    # (B,1,N) @ (B,N,M) -> (B,1,M)
    Fr = torch.bmm(x, cos).squeeze(1) / max(N, 1)      # (B,M)
    Fi = torch.bmm(x, -sin).squeeze(1) / max(N, 1)     # (B,M)
    return Fr, Fi

def _band_weight_schedule(
    kappa: torch.Tensor,
    *,
    split_fracs: Tuple[float, float] = (0.4, 0.8),
    global_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    |kappa| 분위수로 저/중/고 밴드를 나눠 밴드별 스케줄 가중치 생성.
    초반(low-heavy) → 후반(high-heavy) 선형 보간. 최종 평균이 1이 되도록 정규화.
    반환: w_k (M,)
    """
    k_abs = kappa.abs()
    q1, q2 = split_fracs
    try:
        th1 = torch.quantile(k_abs, q1)
        th2 = torch.quantile(k_abs, q2)
    except Exception:
        k_sorted, _ = torch.sort(k_abs)
        i1 = int((len(k_sorted) - 1) * q1)
        i2 = int((len(k_sorted) - 1) * q2)
        th1, th2 = k_sorted[i1], k_sorted[i2]

    # 밴드 마스크
    low  = k_abs <= th1
    mid  = (k_abs > th1) & (k_abs <= th2)
    high = k_abs > th2

    # 진행도 s ∈ [0,1]
    if (global_step is None) or (max_steps is None) or max_steps <= 0:
        s = 0.0
    else:
        s = float(global_step) / float(max_steps)
        s = max(0.0, min(1.0, s))

    # 시작/끝 밴드 가중치(필요시 조정 가능)
    low_start,  mid_start,  high_start  = 1.00, 0.30, 0.10
    low_final,  mid_final,  high_final  = 0.20, 0.80, 1.00

    wL = (1.0 - s) * low_start  + s * low_final
    wM = (1.0 - s) * mid_start  + s * mid_final
    wH = (1.0 - s) * high_start + s * high_final

    w = torch.empty_like(kappa)
    w[low]  = wL
    w[mid]  = wM
    w[high] = wH

    # 평균 1로 정규화(훈련 내내 스케일 안정)
    w = w * (w.numel() / (w.sum() + 1e-12))
    return w

def hilbert_loss_fn(
    model,
    sde,
    x_0: torch.Tensor,                # (B,N) 정규화된 타깃
    t: torch.Tensor,                  # (B,)
    e: torch.Tensor,                  # (B,N) 힐버트 노이즈
    x_coord: torch.Tensor,            # (B,N) 정규화 좌표 [-1,1]
    *,
    global_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    (1) 좌표 적분형 MSE(기존 힐버트 손실)
    (2) NUDFT 기반 스펙트럼 가중 잔차 손실(옵션)
        L_spec = Σ_k w_k |r~(k)|^2,  r = (output - target)
    """
    # ----- 1) 표준(좌표 가중) 스코어 손실 -----
    x_mean = sde.diffusion_coeff(t)               # ᾱ(t)
    noise  = sde.marginal_std(t)                  # σ(t)
    x_t    = x_0 * x_mean[:, None] + e * noise[:, None]
    target = -e

    model_input = torch.cat([x_t.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)
    output = model(model_input, t.float())        # (B,N) — score(x_t)

    # trapezoid 가중 평균으로 스케일 보존
    w_trapz = _trapz_weights(x_coord)
    w_trapz = w_trapz / (w_trapz.sum(dim=1, keepdim=True) + 1e-12)
    data_loss = ((output - target).square() * w_trapz).sum(dim=1).mean(dim=0)

    return data_loss

