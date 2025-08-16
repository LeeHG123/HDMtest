import torch
import torch.nn.functional as F

def _trapezoidal_weights(x_coord: torch.Tensor):
    """
    x_coord: (B, N) 또는 (B, N, 1)
    반환
      w      : (B, N)  - 사다리꼴 규칙 가중치 (비음수, 합은 도메인 길이)
      sortix : (B, N)  - 각 배치에서 오름차순 정렬 인덱스
    정렬된 좌표 기준으로 w를 만들고, 이후 예측/타깃을 같은 순서로 gather해서 씁니다.
    """
    if x_coord.dim() == 3:
        x = x_coord.squeeze(-1)
    else:
        x = x_coord
    # 각 배치별 좌표 정렬
    xs, sortix = torch.sort(x, dim=1)
    # 인접 간격
    dx = xs[:, 1:] - xs[:, :-1]
    # 사다리꼴 가중치
    w = torch.zeros_like(xs)
    w[:, 1:-1] = 0.5 * (dx[:, 1:] + dx[:, :-1])
    w[:, 0]    = 0.5 * dx[:, 0]
    w[:, -1]   = 0.5 * dx[:, -1]
    # 수치 안전장치
    w = torch.clamp(w, min=0.0)
    return w, sortix

def loss_fn(model, sde, x_0, t, e):
    """
    (기존) 좌표가 모델 입력에 포함되지 않는 베이스라인.
    균등격자 가정이므로 그대로 두되, 호환을 위해 유지합니다.
    """
    x_mean = sde.diffusion_coeff(x_0, t)
    noise  = sde.marginal_std(e, t)
    x_t    = x_mean + noise
    score  = -noise
    output = model(x_t, t)
    loss   = (output - score).square().sum(dim=(1,2,3)).mean(dim=0)
    return loss

def hilbert_loss_fn(
    model,
    sde,
    x_0: torch.Tensor,     # (B, N)
    t:   torch.Tensor,     # (B,)
    e:   torch.Tensor,     # (B, N)   - Hilbert noise sample
    x_coord: torch.Tensor, # (B, N)
    lambda_norm: float = 0.1,
):
    """
    비균등 격자 지원 손실:
      L = L2_weighted + lambda_norm * Normalization_weighted
    - L2_weighted = ( Σ_i w_i (ŷ_i - y_i)^2 ) / ( Σ_i w_i )
    - Normalization_weighted = ( (Σ_i w_i ŷ_i - Σ_i w_i y_i)^2 ) / (Σ_i w_i)^2
    여기서 y = score = -e,  ŷ = model(x_t, t)
    """
    # 1) forward noising
    x_mean = sde.diffusion_coeff(t)           # ᾱ(t)
    noise  = sde.marginal_std(t)              # σ(t)
    x_t    = x_0 * x_mean[:, None] + e * noise.view(-1, 1)
    target = -e                                # score target (ε-parameterization)

    # 2) 모델 입력: [신호, 좌표] 2채널
    model_input = torch.cat([x_t.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)  # (B, 2, N)
    pred = model(model_input, t.float())                                      # (B, N)

    # 3) 좌표 기반 가중치 (사다리꼴)
    w, sortix = _trapezoidal_weights(x_coord)          # (B, N), (B, N)
    wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-12) # 도메인 길이 L

    # 4) 예측/타깃을 같은 정렬 순서로 정렬해 가중합 계산
    resid_sorted  = torch.gather(pred - target, 1, sortix)     # (B, N)
    pred_sorted   = torch.gather(pred,         1, sortix)
    target_sorted = torch.gather(target,       1, sortix)

    # 4-a) 가중 L2
    l2_per_sample = (w * resid_sorted.square()).sum(dim=1, keepdim=True) / wsum
    l2_loss = l2_per_sample.mean()

    # 4-b) 정규화(적분 일치) 항
    int_pred   = (w * pred_sorted).sum(dim=1, keepdim=True)     # ∑ w_i ŷ_i
    int_target = (w * target_sorted).sum(dim=1, keepdim=True)   # ∑ w_i y_i
    norm_per_sample = (int_pred - int_target).square() / (wsum.square())
    norm_loss = norm_per_sample.mean()

    total = l2_loss + lambda_norm * norm_loss
    return total
