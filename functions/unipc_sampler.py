from typing import Optional,Tuple
import torch, tqdm
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from functions.sde import VPSDE1D


def make_unipc_scheduler(
        sde: VPSDE1D,                    
        num_train_steps: int = 1_000,
        device: str | torch.device = "cuda",
        *,                           
        solver_order: int = 4,       #   2·3·4 지원
        lower_order_final: bool = True,
) -> UniPCMultistepScheduler:
    """
    HDM VPSDE1D와 수치적으로 동일한 ᾱ(t), β(t) 이산화 테이블을 갖는
    UniPC-Multistep Scheduler를 생성한다.
    ----------
    - t_i = (i+1)/N · T  (i=0,…,N−1)
    - ᾱ₀ = 1,  β_i = 1 − ᾱ(t_i)/ᾱ(t_{i−1})
    """
    T = sde.T                              # 0.9946  (cosine schedule)

    # ──────────────────────────────────────────────
    # 0 ≤ t₀ < … < t_N = T  (길이 N+1)
    # β_i := 1 − ᾱ(t_i)/ᾱ(t_{i−1})   (i = 1…N)
    # ──────────────────────────────────────────────
    ts = torch.linspace(
        0.0, T, num_train_steps + 1,         # shape (N+1,)
        dtype=torch.float64, device=device
    )

    log_alpha = sde.marginal_log_mean_coeff(ts)          # ln ᾱ(t)
    alpha_bar = torch.exp(log_alpha)                     # ᾱ(t)

    alpha_bar_t = alpha_bar[1:]
    alpha_bar_t_minus_1 = alpha_bar[:-1]
    
    # beta_t = 1 - (alpha_bar_t / alpha_bar_t-1)
    # 분모가 0이 되는 것을 방지하고, 비율이 1을 넘지 않도록 clamp
    beta_t = 1.0 - torch.clamp(alpha_bar_t / alpha_bar_t_minus_1, max=0.9999)
    
    # 최종 betas 값도 안정적인 범위 내로 클램핑
    betas = torch.clamp(beta_t, 0.0, 0.9999)

    # diffusers 는 CPU tensor/numpy 를 기대 → .cpu()
    scheduler = UniPCMultistepScheduler(
        trained_betas       = betas.float().cpu(),
        num_train_timesteps = len(betas),   # = N
        solver_type         = "bh2",        # ODE 모드
        solver_order        = solver_order, # ★ 3 또는 4 차
        lower_order_final   = lower_order_final,
        prediction_type     = "epsilon",
        thresholding        = False,
        sample_max_value    = 1.0,
    )  
    return scheduler


@torch.no_grad()
def sample_probability_flow_ode(
        model,
        sde: VPSDE1D,
        x_t0: torch.Tensor, 
        x_coord: torch.Tensor,       
        batch_size: Optional[int] = None,
        data_dim: Optional[int] = None,
        device: torch.device = torch.device("cuda"),
        inference_steps: int = 500,
        fp16: bool = False,
        progress: bool = True, 
):
    """
    Args
    ----
    model            : score network,  (B, N) → (B, N)
    sde              : VPSDE1D (시간 스케일 T=0.9946 기준)
    inference_steps  : NFE (= scheduler steps)
    x_t0             : None 이면 N(0, I) 에서 샘플링
    Returns
    -------
    x_0 (torch.Tensor) : (B, data_dim) 복원 샘플
    """

    scheduler = make_unipc_scheduler(
        sde,
        num_train_steps=1000,
        device=device,
        solver_order=4,              # 필요 시 4로 조정
        lower_order_final=True,
    )
    scheduler.set_timesteps(inference_steps, device=device)

    # 초기분포: p_T = 𝒩(0, σ_T² I)
    if x_t0 is None:
        if batch_size is None or data_dim is None:
            raise ValueError("batch_size and data_dim must be provided if x_t0 is not given.")
        sigma_target = 1.0
        x = torch.randn(batch_size, data_dim, device=device) * sigma_target
    else:
        x = x_t0.to(device)
        x_coord = x_coord.to(device)
        # x_t0에서 shape 정보를 추론
        batch_size, data_dim = x.shape

    model.eval()
    autocast = torch.cuda.amp.autocast if fp16 else torch.no_grad

    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps, disable=not progress)):
        with autocast():
            #   scheduler.timesteps → {N-1, …, 0}
            #   실제 시각   tᵢ = (i+1)/N · T  (cosine VPSDE)
            t_cont = t.to(torch.float32) / scheduler.config.num_train_timesteps * sde.T
            t_vec  = t_cont.repeat(batch_size).to(device).to(x.dtype)       # shape (B,)
            model_input = torch.cat([x.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)
            score  = model(model_input, t_vec)                      # score(x, t_cont)

        # diffusers expects epsilon → ε = - score
        sigma_t = scheduler.sigmas.to(device)[i]   
        epsilon = - score
        x = scheduler.step(epsilon, t, x).prev_sample

        flat  = x.view(x.size(0), -1)            # (B, N) 로 펴기
        norm  = flat.norm(dim=1, keepdim=True)   # 각 샘플 L2-norm
        mask = (norm > 10.0).squeeze(1)          # Quadratic 은 임계 10 
        flat[mask] = flat[mask] / norm[mask] * 10.0
        x = flat.view_as(x)

    return x

