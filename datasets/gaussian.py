# datasets/gaussian.py
import torch
import math
from datasets.adapted_grid import (
    build_adaptive_grid_1d,
    build_adapted_random_distribution_1d,
    sample_adapted_random_points_1d,
)

def _subgroup_meta_from_grid(x_base: torch.Tensor, G: int):
    x_min = float(x_base.min().item())
    x_max = float(x_base.max().item())
    bounds = torch.linspace(x_min, x_max, steps=G+1)
    counts = torch.zeros(G, dtype=torch.long)
    for g in range(G):
        a = bounds[g].item(); b = bounds[g+1].item()
        if g < G-1:
            mask = (x_base >= a) & (x_base < b)
        else:
            mask = (x_base >= a) & (x_base <= b)
        counts[g] = int(mask.sum().item())
    return {"subgroups": G, "bounds": bounds, "counts": counts}

class GaussianDataset(torch.utils.data.Dataset):
    """
    y = log φ(x) + ε 를 z-score 정규화하여 저장
    역변환: exp(y_log)
    """
    def __init__(self, num_data: int, num_points: int, seed: int = 42,
                 grid_type: str = 'uniform', subgroups: int = 3,
                 adapt_params: dict | None = None,
                 noise_std: float = 0.0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_data = num_data
        self.num_points = num_points
        self.seed = seed
        self.grid_type = grid_type
        self.noise_std  = float(noise_std)
        self.is_train = True
        self.noise_std = float(noise_std)

        # 좌표 정규화 파라미터(러너와 일치)
        self.coord_scale  = 10.0
        self.coord_offset = 0.0

        def base_func(x: torch.Tensor) -> torch.Tensor:
            return -0.5 * x**2 - 0.5 * math.log(2 * math.pi)

        if grid_type == 'uniform':
            x_base = torch.linspace(-10., 10., steps=num_points)
            meta = _subgroup_meta_from_grid(x_base, int(subgroups))

        elif grid_type == 'random':
            x_base = (torch.rand(num_points) * 20 - 10).sort().values
            meta = _subgroup_meta_from_grid(x_base, int(subgroups))

        elif grid_type == 'adapted':
            ap = {} if adapt_params is None else dict(adapt_params)
            x_base, meta = build_adaptive_grid_1d(
                x_min=-10.0, x_max=10.0, num_points=self.num_points,
                subgroups=int(subgroups),
                func=base_func,
                pilot_points=int(ap.get('pilot_points', 2048)),
                alpha=float(ap.get('alpha', 1.0)),
                smooth_sigma=float(ap.get('smooth_sigma', 3.0)),
                clip_quantile=float(ap.get('clip_quantile', 0.98)),
                min_per_group=int(ap.get('min_per_group', 1)),
                device=torch.device('cpu'),
                dtype=torch.float32,
            )

        elif grid_type == 'adapted_random':
            ap = {} if adapt_params is None else dict(adapt_params)
            self.subgroups = int(subgroups)
            self._adapt = ap
            self._adapt_state = build_adapted_random_distribution_1d(
                x_min=-10.0, x_max=10.0, subgroups=self.subgroups,
                func=base_func,
                pilot_points=int(ap.get('pilot_points', 2048)),
                alpha=float(ap.get('alpha', 1.0)),
                smooth_sigma=float(ap.get('smooth_sigma', 3.0)),
                clip_quantile=float(ap.get('clip_quantile', 0.98)),
                uniform_mix=float(ap.get('uniform_mix', 0.15)),
                device=torch.device('cpu'), dtype=torch.float64,
            )
            x_base, meta = sample_adapted_random_points_1d(
                self._adapt_state, num_points=self.num_points,
                min_per_group=int(ap.get('min_per_group', 1)),
                alpha_for_allocation=float(ap.get('alpha', 1.0)),
            )

        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'.")

        self.subgroup_meta = meta
        self.x = x_base.unsqueeze(0).repeat(num_data, 1)

        # 데이터 생성 및 정규화 통계
        log_phi = -0.5 * self.x**2 - 0.5 * math.log(2 * math.pi)
        eps = torch.randn_like(log_phi) * self.noise_std
        log_phi_noisy = log_phi + eps

        self.mean = log_phi_noisy.mean()
        self.std  = log_phi_noisy.std().clamp_min(1e-8)
        self.dataset = (log_phi_noisy - self.mean) / self.std

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        # grid_type == 'random' : 매 호출마다 균등무작위 좌표
        if self.grid_type == 'random' and getattr(self, 'is_train', False):
            x_item = (torch.rand(self.num_points) * 20 - 10).sort().values
            log_phi = -0.5 * x_item**2 - 0.5 * math.log(2 * math.pi)
            eps = torch.randn_like(log_phi) * self.noise_std
            log_phi_noisy = log_phi + eps
            y_item = (log_phi_noisy - self.mean) / self.std
            return x_item.unsqueeze(-1), y_item.unsqueeze(-1)

        # grid_type == 'adapted_random' : 활동도 기반 층화 역변환 샘플
        if self.grid_type == 'adapted_random' and getattr(self, 'is_train', False):
            x_item, _ = sample_adapted_random_points_1d(
                self._adapt_state, num_points=self.num_points,
                min_per_group=int(self._adapt.get('min_per_group', 1)),
                alpha_for_allocation=float(self._adapt.get('alpha', 1.0)),
            )
            log_phi = -0.5 * x_item**2 - 0.5 * math.log(2 * math.pi)
            eps = torch.randn_like(log_phi) * self.noise_std
            log_phi_noisy = log_phi + eps
            y_item = (log_phi_noisy - self.mean) / self.std
            return x_item.unsqueeze(-1), y_item.unsqueeze(-1)

        # 고정 그리드 샘플
        return (
            self.x[idx].unsqueeze(-1),
            self.dataset[idx].unsqueeze(-1)
        )

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        y_log = y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)
        return torch.exp(y_log)




