# functions/pfdiff_sampler.py
"""
PFDiff (Training-Free Acceleration of Diffusion Models
Combining Past and Future Scores) – ODE sampler for 1-D VPSDE1D.

논문 Algorithm 1(PFDiff-1) 을 그대로 구현:
  1) 과거 score ε_prev 로 springboard x_tmp (Euler 한 스텝) 계산
  2) 그 지점(t_{k+1})에서 future score ε_next 추정
  3) ε̂_k = (1+α)·ε_prev − α·ε_next  로 결합
  4) ε̂_k 를 UniPC 1-step solver에 주입하여 x_k → x_{k+1}

스케줄러는 기존 UniPC 의 β-테이블을 재사용한다.
"""
from typing import Optional
import torch, tqdm
from functions.sde            import VPSDE1D
from functions.unipc_sampler  import make_unipc_scheduler   # ᾱ(t)·β(t) 재사용


@torch.no_grad()
def sample_probability_flow_ode(
    model,
    sde: VPSDE1D,
    *,
    batch_size: Optional[int] = None,
    data_dim:   Optional[int] = None,
    x_t0: Optional[torch.Tensor] = None,   # resolution-free 초기값
    inference_steps: int = 500,
    device           = "cuda",
    fp16: bool       = False,
    progress: bool   = True,
    alpha:  float    = 0.5,                # 논문 default
):
    # ─── 0. Scheduler ────────────────────────────────────────────────
    scheduler = make_unipc_scheduler(sde, num_train_steps=1_000, device=device)
    scheduler.set_timesteps(inference_steps, device=device)

    # ─── 1. 초기 x_T ─────────────────────────────────────────────────
    if x_t0 is None:
        if batch_size is None or data_dim is None:
            raise ValueError("batch_size, data_dim 필요")
        x = torch.randn(batch_size, data_dim, device=device)
    else:
        x = x_t0.to(device)
        batch_size, data_dim = x.shape

    model.eval()
    autocast = torch.cuda.amp.autocast if fp16 else torch.no_grad

    # ─── 2. 첫 past-score ε_prev 계산 ────────────────────────────────
    t_int   = scheduler.timesteps[0]
    t_cont  = t_int.to(torch.float32) / scheduler.config.num_train_timesteps * sde.T
    t_vec   = t_cont.repeat(batch_size).to(x.dtype).to(device)
    with autocast():
        epsilon_prev = -model(x, t_vec)                  # ε_0

    # ─── 3. 메인 루프 ────────────────────────────────────────────────
    for step_idx in tqdm.tqdm(range(len(scheduler.timesteps)),
                              disable=not progress):
        t_idx      = scheduler.timesteps[step_idx]
        t_cont     = t_idx.to(torch.float32) / scheduler.config.num_train_timesteps * sde.T
        t_vec      = t_cont.repeat(batch_size).to(x.dtype).to(device)

        # 마지막 스텝: future-score 없음 → ε̂ = ε_prev 로 단순 업데이트
        is_last = (step_idx == len(scheduler.timesteps) - 1)
        if is_last:
            epsilon_hat = epsilon_prev
            x = scheduler.step(epsilon_hat, t_idx, x).prev_sample
            break

        # 3-A. springboard: Euler 한 스텝으로 x_tmp (t_{k+1})
        t_idx_next   = scheduler.timesteps[step_idx + 1]
        t_cont_next  = t_idx_next.to(torch.float32) / scheduler.config.num_train_timesteps * sde.T
        dt           = (t_cont_next - t_cont)           # < 0 (역방향)
        beta_t       = sde.beta(t_vec)                  # β(t_k)
        drift        = -0.5 * beta_t[:, None] * (x + (-epsilon_prev))
        x_tmp        = x + dt * drift                   # Euler full step

        # 3-B. future score ε_next at (x_tmp , t_{k+1})
        t_vec_next = t_cont_next.repeat(batch_size).to(x.dtype).to(device)
        with autocast():
            epsilon_next = -model(x_tmp, t_vec_next)

        # 3-C. 결합 ε̂_k
        epsilon_hat = (1.0 + alpha) * epsilon_prev - alpha * epsilon_next

        # 3-D. UniPC 1-스텝 advance  (x_k → x_{k+1})
        x = scheduler.step(epsilon_hat, t_idx, x).prev_sample

        # 3-E. norm-clamp (Quadratic 전용 규칙)
        flat = x.view(batch_size, -1)
        norm = flat.norm(dim=1, keepdim=True)
        mask = (norm > 10).squeeze(1)
        flat[mask] = flat[mask] / norm[mask] * 10
        x = flat.view_as(x)

        # 3-F. 버퍼 갱신
        epsilon_prev = epsilon_next.detach()            # 다음 step의 past-score

    return x

