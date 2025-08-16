# functions/dpmpp_sampler.py
from typing import Optional
import torch, tqdm
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from functions.sde import VPSDE1D

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
    해상도-의존적 벡터 L2 노름 대신, 연속영역 RMS를 사용해 클립한다.
    sqrt( (1/L) ∫ x(z)^2 dz ) <= thr 를 강제.
    """
    if not torch.isfinite(torch.tensor(thr)) or thr <= 0:
        return x
    w = _trapz_weights(x_coord)                                   # (B, N)
    L = w.sum(dim=1, keepdim=True).clamp_min(1e-12)               # (B, 1)
    rms = torch.sqrt((w * x.pow(2)).sum(dim=1, keepdim=True) / L) # (B, 1)
    mask = (rms > thr).squeeze(1)                                 # (B,)
    if mask.any():
        x = x.clone()
        x[mask] = x[mask] * (thr / rms[mask])
    return x

def make_dpmpp_scheduler(
    sde: VPSDE1D,
    num_train_steps: int = 1_000,
    device: torch.device | str = "cuda",
    *,
    solver_order: int = 3,          # DPMSolver++ 권장: 3(무조건) / 2(조건)
    solver_type: str = "bh2",       # 확률-흐름 ODE 전용
    lower_order_final: bool = True,
) -> DPMSolverMultistepScheduler:
    """
    HDM VPSDE1D와 동일한 ᾱ(t)·β(t) 테이블을 갖는
    DPMSolver++ Multistep Scheduler 생성.
    """
    T = sde.T
    ts = torch.linspace(
        0.0, T, num_train_steps + 1,
        dtype=torch.float64, device=device,
    )
    log_alpha = sde.marginal_log_mean_coeff(ts)
    alpha_bar = torch.exp(log_alpha)

    betas = 1.0 - torch.clamp(alpha_bar[1:] / alpha_bar[:-1], max=0.9999)
    betas = torch.clamp(betas, 0.0, 0.9999).float().cpu()

    return DPMSolverMultistepScheduler(
        trained_betas       = betas,
        num_train_timesteps = len(betas),
        algorithm_type      = "dpmsolver++",   #★  DPM-Solver++
        solver_order        = solver_order,
        solver_type         = solver_type,
        lower_order_final   = lower_order_final,
        prediction_type     = "epsilon",
        thresholding        = False,
        sample_max_value    = 1.0,
    )


@torch.no_grad()
def sample_probability_flow_ode(
    model,
    sde: VPSDE1D,
    x_t0: torch.Tensor,      
    x_coord: torch.Tensor,      
    *,
    batch_size: Optional[int] = None,
    data_dim: Optional[int] = None,
    device: torch.device | str = "cuda",
    inference_steps: int = 500,
    fp16: bool = False,
    progress: bool = True,
    enable_rms_clip: bool = False,
    rms_clip_threshold: Optional[float] = None,    
):
    """DPMSolver++ ODE (확률-흐름) 샘플링."""
    scheduler = make_dpmpp_scheduler(sde, device=device)
    scheduler.set_timesteps(inference_steps, device=device)

    if x_t0 is None:
        if batch_size is None or data_dim is None:
            raise ValueError("batch_size·data_dim 필요")
        x = torch.randn(batch_size, data_dim, device=device)
    else:
        x = x_t0.to(device)
        batch_size, data_dim = x.shape

    model.eval()
    autocast = torch.cuda.amp.autocast if fp16 else torch.no_grad

    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps, disable=not progress)):
        with autocast():
            t_cont = t.to(torch.float32) / scheduler.config.num_train_timesteps * sde.T
            t_vec  = t_cont.repeat(batch_size).to(device).to(x.dtype)
            model_input = torch.cat([x.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)
            score  = model(model_input, t_vec)                        # ∇ₓ log p(xᵗ)
        epsilon = -score                                    # prediction_type="epsilon"
        x = scheduler.step(epsilon, t, x).prev_sample

        #flat  = x.view(x.size(0), -1)            # (B, N) 로 펴기
        #norm  = flat.norm(dim=1, keepdim=True)   # 각 샘플 L2-norm
        #mask = (norm > 10.0).squeeze(1)          # Quadratic 은 임계 10 
        #flat[mask] = flat[mask] / norm[mask] * 10.0
        #x = flat.view_as(x)    
    if enable_rms_clip and (rms_clip_threshold is not None):
        x = rms_clip_resolution_free(x, x_coord, float(rms_clip_threshold))                

    return x
