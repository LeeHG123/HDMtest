import torch
import torchode as to
from functions.sde import VPSDE1D

@torch.no_grad()
def sample_probability_flow_ode(
        model,
        sde: VPSDE1D,
        *,
        x_t0:      torch.Tensor,      # (B, N) 초기 상태 (x_T)
        x_coord:   torch.Tensor,
        inference_steps: int = 500,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        device="cuda",
):
    """
    resolution-free 확률-흐름 ODE 샘플러 ― Tsitouras 5(4)+IntegralController.
    - 역시간 적분을 위해 변수 변환(s = T - t) 적용
    - 모델 출력(noise)을 실제 스코어(score)로 스케일링 (output / std_t)
    """
    model.eval()
    
    x = x_t0.to(device)
    batch, dim = x.shape
    
    T = sde.T
    eps = sde.eps

    # 1. ODE 우변(drift) 정의: dz/ds = -f(t, z)
    def reverse_f(s, y):
        # 솔버의 시간 s를 확산 시간 t로 변환
        t = T - s + eps
        t_vec = t.expand(batch)
        
        # 현재 상태 y와 좌표 x_coord를 결합
        model_input = torch.cat([y.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)
        
        # 모델 출력은 noise `-e`를 근사
        model_output = model(model_input, t_vec)
        
        # 시간 t에서의 표준편차(노이즈 스케일) 계산
        std_t = sde.marginal_std(t_vec)
        
        # 모델 출력을 실제 스코어(∇log p)로 변환: score = -e / σ(t)
        score_true = model_output / std_t[:, None]
        
        # 순방향 ODE drift 계산
        beta_t = sde.beta(t_vec)
        forward_drift = -0.5 * beta_t[:, None] * (y + score_true)
        
        # 변수 변환에 따라 부호 반전
        return -forward_drift

    # 2. ODE 솔버 설정
    term   = to.ODETerm(reverse_f)
    step   = to.Tsit5(term=term)      # step   = to.Dopri5(term=term) 이면 Dopri5 사용. 이때, hilbert_runner에서 visualization 이름 바꿔야 함.
    ctrl   = to.IntegralController(atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(step, ctrl)

    # 3. 시간 스케일 정의 (증가하는 방향)
    s_eval = torch.linspace(eps, T, inference_steps + 1, device=device).expand(batch, -1)

    # 4. 초기 값 문제(IVP) 정의
    problem = to.InitialValueProblem(y0=x, t_eval=s_eval)

    # 5. 적분 수행
    sol = solver.solve(problem)
    x0  = sol.ys[:, -1]

    # Quadratic 데이터셋 norm-clamp
    flat = x0.view(batch, -1)
    norm = flat.norm(dim=1, keepdim=True)
    mask = (norm > 10.0).squeeze(1)          
    flat[mask] = flat[mask] / norm[mask] * 10.0
    return flat.view_as(x0)