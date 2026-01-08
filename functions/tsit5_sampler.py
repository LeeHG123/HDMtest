import torch
import torchode as to
from typing import Optional
from functions.sde import VPSDE1D
# -----------------------------------------------------------------------------
# 해상도-불변 RMS 클립: 연속영역 L2 평균을 사다리꼴 적분으로 근사
# -----------------------------------------------------------------------------
def _trapz_weights(x_coord: torch.Tensor) -> torch.Tensor:
    """
    x_coord: (B, N)
    비등간격 좌표도 다루기 위해 배치별로 정렬된 좌표 기준의 사다리꼴 가중치를 만든다.
    """
    xs, _ = torch.sort(x_coord, dim=1)           # (B, N)
    if xs.size(1) == 1:
        return torch.ones_like(xs)

    dx = xs[:, 1:] - xs[:, :-1]                  # (B, N-1)
    w = torch.zeros_like(xs)                     # (B, N)
    w[:, 0] = 0.5 * dx[:, 0]
    w[:, -1] = 0.5 * dx[:, -1]
    if xs.size(1) > 2:
        w[:, 1:-1] = 0.5 * (dx[:, 1:] + dx[:, :-1])
    return w.clamp_min(0.0)


def rms_clip_resolution_free(
    x: torch.Tensor, x_coord: torch.Tensor, thr: float
) -> torch.Tensor:
    """
    x: (B, N)
    x_coord: (B, N)
    thr: RMS 상한. sqrt( (1/L) ∫ x(z)^2 dz ) <= thr 를 강제

    해상도-의존적 벡터 L2 노름 대신, 연속영역 RMS를 사용하여
    res_free_points 변화에도 진폭이 일정하도록 만든다.
    """
    if not torch.isfinite(torch.tensor(thr)) or thr <= 0:
        return x

    w = _trapz_weights(x_coord)                                   # (B, N)
    L = w.sum(dim=1, keepdim=True).clamp_min(1e-12)               # (B, 1)
    rms = torch.sqrt((w * x.pow(2)).sum(dim=1, keepdim=True) / L) # (B, 1)

    mask = (rms > thr).squeeze(1)                                 # (B,)
    if mask.any():
        x_scaled = x[mask] * (thr / rms[mask])
        x = x.clone()
        x[mask] = x_scaled
    return x


# -----------------------------------------------------------------------------
# Tsitouras 5(4) 확률-흐름 ODE 샘플러 (resolution-free)
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_probability_flow_ode(
    model,
    sde: VPSDE1D,
    *,
    x_t0: torch.Tensor,                 # (B, N) 초기 상태: x_T
    x_coord: torch.Tensor,              # (B, N) 좌표
    inference_steps: int = 500,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    device: str = "cuda",
    # 선택: 발산 방지용 해상도-불변 RMS 클립(기본 비활성)
    enable_rms_clip: bool = False,
    rms_clip_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    Resolution-free 확률-흐름 ODE 샘플러 ― Tsitouras 5(4) + IntegralController.

    - 역시간 적분을 위해 변수 변환 s = T - t (단조 증가) 를 사용한다.
    - 모델 출력이 예측 노이즈 -e 를 근사한다고 가정하면,
      score(t, x) = model_out / sigma(t) 로 변환하면 된다.
    - 확률-흐름 ODE (VPSDE):
        dx = [-1/2 * beta(t) * x - 1/2 * beta(t) * score(t, x)] dt
      따라서 reverse time 변수 s 에 대해 dz/ds = -f(t, z).
    - 해상도에 의존해 진폭을 깎는 후처리(L2 클리핑)를 제거했다.
      필요시엔 해상도-불변 RMS 클립을 옵션으로 제공한다.
    """
    model.eval()

    # 장치/형상 정렬
    x = x_t0.to(device)
    x_coord = x_coord.to(device)
    batch, dim = x.shape

    T = sde.T
    eps = sde.eps

    # -----------------------------
    # 1) 역시간 ODE 우변: dz/ds = -f(t, z)
    # -----------------------------
    def reverse_f(s, y):
        # s \in [eps, T]  ->  t = T - s + eps \in [T, eps]
        t = T - s + eps
        t_vec = t.expand(batch)                           # (B,)

        # 모델 입력: (B, 2, N)  [signal, coord]
        model_input = torch.cat([y.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)

        # 모델 출력은 -e_hat 를 근사. score = (-e_hat) / sigma(t) = model_out / sigma(t)
        model_out = model(model_input, t_vec)             # (B, N)
        sigma_t = sde.marginal_std(t_vec)                 # (B,)
        score = model_out / sigma_t[:, None]              # (B, N)

        beta_t = sde.beta(t_vec)                          # (B,)
        # f(t, y) = -1/2 * beta * (y + score)
        forward_drift = -0.5 * beta_t[:, None] * (y + score)

        # dz/ds = - f(t, z)  (s = T - t)
        return -forward_drift

    # -----------------------------
    # 2) ODE 솔버 구성
    # -----------------------------
    term = to.ODETerm(reverse_f)
    step = to.Tsit5(term=term)
    ctrl = to.IntegralController(atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(step, ctrl)

    # -----------------------------
    # 3) 적분 구간 및 IVP 정의
    # -----------------------------
    s_eval = torch.linspace(eps, T, inference_steps + 1, device=device).expand(batch, -1)
    problem = to.InitialValueProblem(y0=x, t_eval=s_eval)

    # -----------------------------
    # 4) 적분 실행
    # -----------------------------
    sol = solver.solve(problem)
    x0 = sol.ys[:, -1]                                     # (B, N)

    # -----------------------------
    # 5) (선택) 해상도-불변 RMS 클립
    # -----------------------------
    if enable_rms_clip and (rms_clip_threshold is not None):
        x0 = rms_clip_resolution_free(x0, x_coord, float(rms_clip_threshold))

    # 해상도 의존적인 전역 L2 클리핑은 적용하지 않는다.
    return x0