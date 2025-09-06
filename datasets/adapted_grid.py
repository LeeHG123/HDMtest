import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def _gaussian_kernel1d(sigma: float, dtype=torch.float64):
    if sigma <= 0:
        return None
    radius = int(math.ceil(3.0 * float(sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=dtype)
    ker = torch.exp(-0.5 * (xs / float(sigma)) ** 2)
    ker = ker / (ker.sum() + 1e-12)
    return ker

@torch.no_grad()
def _smooth1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    ker = _gaussian_kernel1d(sigma, dtype=x.dtype)
    if ker is None:
        return x
    ker = ker.view(1, 1, -1)
    pad = (ker.size(-1) - 1) // 2
    xx = x.view(1, 1, -1)
    xx = F.pad(xx, (pad, pad), mode='reflect')
    y = F.conv1d(xx, ker)
    return y.view(-1)

@torch.no_grad()
def _central_diff(y: torch.Tensor, dx: float) -> torch.Tensor:
    L = y.numel()
    if L < 2:
        return torch.zeros_like(y)
    d = torch.empty_like(y)
    d[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
    d[0]    = (y[1] - y[0]) / dx
    d[-1]   = (y[-1] - y[-2]) / dx
    return d

@torch.no_grad()
def _proportional_allocation(weights, total: int, min_per_group: int = 1):
    G = int(len(weights))
    if G <= 0:
        return []
    w = torch.as_tensor(weights, dtype=torch.float64)
    w = torch.clamp(w, min=0.0)
    base = [int(min_per_group)] * G
    remain = total - sum(base)
    if remain <= 0:
        k = total
        out = [0] * G
        for i in range(G):
            t = min(base[i], k)
            out[i] = int(t)
            k -= t
        return out

    s = float(w.sum())
    if s <= 0.0:
        q = remain // G
        r = remain - q * G
        for i in range(G):
            base[i] += q + (1 if i < r else 0)
        return base

    raw = w / s * remain
    floor = torch.floor(raw).to(torch.int64)
    out = [base[i] + int(floor[i].item()) for i in range(G)]
    leftover = remain - int(floor.sum().item())
    if leftover > 0:
        frac = (raw - floor.to(raw.dtype)).tolist()
        idx = sorted(range(G), key=lambda i: frac[i], reverse=True)
        for k in range(leftover):
            out[idx[k]] += 1
    return out

@torch.no_grad()
def build_adaptive_grid_1d(
    *,
    x_min: float,
    x_max: float,
    num_points: int,
    subgroups: int,
    func,                       # callable(x: Tensor[L]) -> Tensor[L]
    pilot_points: int = 2048,
    alpha: float = 1.0,
    smooth_sigma: float = 3.0,
    clip_quantile: float = 0.98,
    min_per_group: int = 1,
    device: torch.device | None = None,
    dtype=torch.float32,
):
    """
    데이터 종속적이지 않은 1D 적응형 그리드 생성기.
    반환: (x: (N,), meta: {'subgroups','bounds','counts'})
    """
    if device is None:
        device = torch.device('cpu')
    x_min = float(x_min); x_max = float(x_max)
    assert num_points >= subgroups >= 1

    # 1) 활동도 평가
    P = int(max(pilot_points, num_points))
    x_p = torch.linspace(x_min, x_max, steps=P, device=device, dtype=torch.float64)
    with torch.no_grad():
        y_p = func(x_p).to(torch.float64)
        dx = (x_max - x_min) / max(P - 1, 1)
        dy = _central_diff(y_p, dx).abs()
        if smooth_sigma > 0:
            dy = _smooth1d(dy, float(smooth_sigma))
        if 0.0 < float(clip_quantile) < 1.0:
            q = torch.quantile(dy, float(clip_quantile))
            dy = torch.clamp(dy, max=float(q))
        activity = dy

    # 2) subgroup 평균 활동도
    G = int(subgroups)
    bounds = torch.linspace(x_min, x_max, steps=G + 1, device=device, dtype=torch.float64)
    rel  = (x_p - x_min) / (x_max - x_min + 1e-12) * G
    idxs = rel.floor().clamp(min=0, max=G - 1).to(torch.int64)  # ← 마지막 그룹도 활성화
    w = torch.zeros(G, dtype=torch.float64, device=device)
    for g in range(G):
        mask = (idxs == g)
        w[g] = activity[mask].mean() if mask.any() else 0.0
    weights = torch.pow(1e-8 + w, float(alpha))

    # 3) 할당 수 결정
    counts = _proportional_allocation(weights, total=int(num_points), min_per_group=int(min_per_group))

    # 4) 구간별 그리드 (총합 n 유지)
    xs = []
    for g in range(G):
        a = float(bounds[g].item())
        b = float(bounds[g+1].item())
        n = int(counts[g])
        if n <= 0:
            continue
        if g == 0:
            xt = torch.tensor([(a + b) * 0.5], dtype=torch.float64, device=device) if n == 1 \
                 else torch.linspace(a, b, steps=n, dtype=torch.float64, device=device)
        else:
            xt = torch.tensor([(a + b) * 0.5], dtype=torch.float64, device=device) if n == 1 \
                 else torch.linspace(a, b, steps=n + 1, dtype=torch.float64, device=device)[1:]
        xs.append(xt)

    x = torch.cat(xs, dim=0)
    # 안전장치: 요청한 포인트 수와 정확히 일치
    assert x.numel() == int(num_points), f"adaptive grid produced {x.numel()} points, expected {num_points}"

    meta = {"subgroups": G, "bounds": bounds.clone().cpu(), "counts": torch.tensor(counts, dtype=torch.long).cpu()}
    return x.to(dtype=dtype), meta



