import torch
import torch.nn.functional as F

def loss_fn(model, sde, x_0, t, e):
    x_mean = sde.diffusion_coeff(x_0, t)
    noise = sde.marginal_std(e, t)

    x_t = x_mean + noise
    score = -noise

    output = model(x_t, t)

    loss = (output - score).square().sum(dim=(1,2,3)).mean(dim=0)
    return loss

def hilbert_loss_fn(model, sde, x_0, t, e, x_coord):
    x_mean = sde.diffusion_coeff(t)
    noise = sde.marginal_std(t)

    x_t = x_0 * x_mean[:, None] + e * noise.view(-1, 1)
    score = -e

    # 모델 입력을 [x_t, x_coord] 2채널로 구성
    model_input = torch.cat([x_t.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)
    output = model(model_input, t.float())

    # L² loss 계산
    loss = (output - score).square().sum(dim=(1)).mean(dim=0)
    return loss

