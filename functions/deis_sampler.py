# functions/deis_sampler.py
from typing import Optional
import torch, tqdm
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from functions.sde import VPSDE1D


# ─────────────────────────────────────────────────────────────
# ①  Scheduler 생성기
# ─────────────────────────────────────────────────────────────
def make_deis_scheduler(
    sde: VPSDE1D,
    *,
    num_train_steps: int = 1_000,
    solver_order: int = 3,            # 2·3 허용, 확률-흐름 ODE 권장: 3
    lower_order_final: bool = True,
    device: torch.device | str = "cuda",
) -> DEISMultistepScheduler:
    """
    HDM VPSDE1D 와 동일한 ᾱ(t)·β(t) 이산 테이블을 공유하는
    DEIS-Multistep Scheduler 생성 함수.
    """
    T = sde.T
    ts = torch.linspace(0.0, T, num_train_steps + 1,
                        dtype=torch.float64, device=device)

    log_alpha = sde.marginal_log_mean_coeff(ts)
    alpha_bar = torch.exp(log_alpha)

    betas = 1.0 - torch.clamp(alpha_bar[1:] / alpha_bar[:-1], max=0.9999)
    betas = torch.clamp(betas, 0.0, 0.9999).float().cpu()

    return DEISMultistepScheduler(
        trained_betas       = betas,
        num_train_timesteps = len(betas),
        prediction_type     = "epsilon",      # Score ⇒ ε 로 변환
        solver_order        = solver_order,
        lower_order_final   = lower_order_final,
        thresholding        = False,
        sample_max_value    = 1.0,
    )


# ─────────────────────────────────────────────────────────────
# ②  확률-흐름 ODE 샘플러 (resolution-free)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_probability_flow_ode(
    model,
    sde: VPSDE1D,
    x_t0: torch.Tensor,       
    x_coord: torch.Tensor,    
    *,
    batch_size: Optional[int] = None,
    data_dim:  Optional[int] = None,
    inference_steps: int = 500,
    device: torch.device | str = "cuda",
    fp16: bool = False,
    progress: bool = True,
):
    """
    DEIS-Multistep 확률-흐름 ODE 샘플링.

    - x_t0 == None : N(0,I)  →  통상적 샘플링
    - x_t0 제공   : resolution-free 초기분포 (Hilbert noise etc.)
    """
    scheduler = make_deis_scheduler(sde, device=device)
    scheduler.set_timesteps(inference_steps, device=device)

    if x_t0 is None:
        if batch_size is None or data_dim is None:
            raise ValueError("batch_size, data_dim must be given when x_t0 is None")
        x = torch.randn(batch_size, data_dim, device=device)
    else:
        x = x_t0.to(device)
        batch_size, data_dim = x.shape

    model.eval()
    autocast = torch.cuda.amp.autocast if fp16 else torch.no_grad

    # diffusers 의 timesteps 는 {N-1,…,0} (정수)
    for t in tqdm.tqdm(scheduler.timesteps, disable=not progress):
        with autocast():
            # 실수 시각으로 매핑 (0 ~ T)
            t_cont = t.to(torch.float32) / scheduler.config.num_train_timesteps * sde.T
            t_vec  = t_cont.repeat(batch_size).to(x.dtype).to(device)
            model_input = torch.cat([x.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)
            score  = model(model_input, t_vec)           # ∇ₓ log p(xᵗ)
            epsilon = -score                    # prediction_type = "epsilon"

        x = scheduler.step(epsilon, t, x).prev_sample

        # Quadratic 데이터셋: 동일 L² 클리핑
        flat = x.view(x.size(0), -1)
        norm = flat.norm(dim=1, keepdim=True)
        mask = (norm > 10.0).squeeze(1)         
        flat[mask] = flat[mask] / norm[mask] * 10.0
        x = flat.view_as(x)

    return x
