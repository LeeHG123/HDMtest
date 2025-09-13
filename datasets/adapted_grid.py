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

@torch.no_grad()
def _build_activity_profile_1d(func, x_min: float, x_max: float, P: int,
                               smooth_sigma: float, clip_quantile: float,
                               dtype=torch.float64, device=None):
    device = device or torch.device('cpu')
    xp = torch.linspace(x_min, x_max, steps=P, device=device, dtype=dtype)   # (P,)
    yp = func(xp).to(dtype)
    dx = (x_max - x_min) / max(P - 1, 1)
    # 중심차분 + 절댓값
    dy = torch.empty_like(yp)
    dy[1:-1] = (yp[2:] - yp[:-2]) / (2.0 * dx)
    dy[0]    = (yp[1] - yp[0]) / dx
    dy[-1]   = (yp[-1] - yp[-2]) / dx
    act = dy.abs()
    # 평활
    if smooth_sigma > 0:
        ker = _gaussian_kernel1d(float(smooth_sigma), dtype=dtype)
        if ker is not None:
            pad = (ker.numel() - 1) // 2
            z = F.pad(act.view(1,1,-1), (pad,pad), mode='reflect')
            act = F.conv1d(z, ker.view(1,1,-1)).view(-1)
    # 상한 클리핑(극단치 완화)
    if 0.0 < float(clip_quantile) < 1.0:
        q = torch.quantile(act, float(clip_quantile))
        act = torch.clamp(act, max=float(q))
    return xp, act.clamp_min(0.0)

@torch.no_grad()
def build_adapted_random_distribution_1d(
    *, x_min: float, x_max: float, subgroups: int,
    func,                   # callable(x:(P,)->(P,))
    pilot_points: int = 2048,
    alpha: float = 1.0,
    smooth_sigma: float = 3.0,
    clip_quantile: float = 0.98,
    uniform_mix: float = 0.15,
    device=None,
    dtype=torch.float64,
):
    """
    활동도 기반 비균질 확률밀도 p(x)를 구성한 뒤,
    - 전역 CDF
    - 각 subgroup 제한 CDF
    를 사전 계산해 반환합니다.
    """
    device = device or torch.device('cpu')
    G = int(subgroups)
    assert G >= 1
    P = int(max(pilot_points, 256))
    xp, act = _build_activity_profile_1d(func, x_min, x_max, P, smooth_sigma, clip_quantile,
                                         dtype=dtype, device=device)
    # 활동도 → 가변밀도 (alpha 지수)
    w_act = (act + 1e-12).pow(float(alpha))
    w_act = w_act / (w_act.sum() + 1e-12)
    # 균등과 혼합해 과집중 방지
    lam = float(uniform_mix)
    w = (1.0 - lam) * w_act + lam * (torch.ones_like(w_act) / w_act.numel())
    w = w / (w.sum() + 1e-12)

    # 전역 CDF (구간 길이를 반영해 piecewise-constant 밀도를 선형보간)
    cdf = torch.cumsum(w, dim=0)
    cdf = cdf / (cdf[-1] + 1e-12)

    # subgroup 경계 및 마스크
    bounds = torch.linspace(x_min, x_max, steps=G+1, device=device, dtype=dtype)
    idxs = torch.clamp(((xp - x_min)/(x_max-x_min+1e-12)*G).floor().to(torch.long), 0, G-1)

    # 그룹별 부분 CDF (정규화)
    group = []
    for g in range(G):
        mask = (idxs == g)
        if not mask.any():
            # 비어있으면 더미(균등)
            xg = bounds[g:g+2]
            group.append({
                'xp': xg, 'cdf': torch.tensor([0.0,1.0], dtype=dtype, device=device),
                'mask': mask
            })
        else:
            xg = xp[mask]
            wg = w[mask]
            cg = torch.cumsum(wg, dim=0)
            cg = cg / (cg[-1] + 1e-12)
            group.append({'xp': xg, 'cdf': cg, 'mask': mask})

    return {
        'x_min': float(x_min), 'x_max': float(x_max),
        'xp': xp, 'pdf': w, 'cdf': cdf,
        'bounds': bounds, 'groups': group,
        'G': G
    }

@torch.no_grad()
def _inverse_cdf_sample_1d(xp: torch.Tensor, cdf: torch.Tensor, n: int, device=None, dtype=None):
    """
    xp:(M,), cdf:(M,) (단조증가, 0..1)
    균등 u~U(0,1)을 역변환하여 x 샘플 반환. 선형보간.
    """
    device = device or xp.device
    dtype  = dtype or xp.dtype
    if n <= 0:
        return torch.empty(0, device=device, dtype=dtype)

    u = torch.rand(n, device=device, dtype=dtype)
    # searchsorted
    k = torch.searchsorted(cdf, u, right=True).clamp(1, cdf.numel()-1)
    cdf_lo = cdf[k-1]; cdf_hi = cdf[k]
    x_lo   = xp[k-1];   x_hi   = xp[k]
    t = ((u - cdf_lo) / (cdf_hi - cdf_lo + 1e-12)).clamp(0,1)
    xs = x_lo + t * (x_hi - x_lo)
    return xs

@torch.no_grad()
def sample_adapted_random_points_1d(
    dist_state: dict,
    *, num_points: int,
    min_per_group: int = 1,
    alpha_for_allocation: float | None = None,
):
    """
    사전 계산된 분포(dist_state)로부터 무작위 샘플 N개를 뽑되,
    - subgroup별 최소 개수를 보장하고
    - (선택) 그룹 가중치에 별도의 지수(alpha_for_allocation)를 줄 수 있음.
    반환: x:(N,), meta:{'subgroups','bounds','counts'}
    """
    G = int(dist_state['G'])
    N = int(num_points)
    bounds = dist_state['bounds']
    groups = dist_state['groups']

    # 그룹 가중치 = 그룹 내 pdf 합
    wg = torch.stack([ (dist_state['pdf'][g['mask']].sum() if g['mask'].any() else torch.tensor(0., device=bounds.device)) for g in groups ])
    if alpha_for_allocation is not None:
        wg = (wg + 1e-12).pow(float(alpha_for_allocation))
    counts = _proportional_allocation(wg, total=N, min_per_group=int(min_per_group))

    xs = []
    for g in range(G):
        n = int(counts[g])
        if n <= 0:
            continue
        xg = _inverse_cdf_sample_1d(groups[g]['xp'], groups[g]['cdf'], n, device=bounds.device, dtype=bounds.dtype)
        xs.append(xg)
    x = torch.cat(xs, dim=0) if xs else torch.empty(0, device=bounds.device, dtype=bounds.dtype)
    x, _ = torch.sort(x)
    meta = {"subgroups": G, "bounds": bounds.clone().cpu(), "counts": torch.tensor(counts, dtype=torch.long).cpu()}
    return x.to(dtype=torch.float32), meta


