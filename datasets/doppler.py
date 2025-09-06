import torch
import math
from datasets.adapted_grid import build_adaptive_grid_1d

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

class DopplerDataset(torch.utils.data.Dataset):
    """
    g(x) = sqrt(x(1-x)) * sin( (2π*1.05)/(x+0.05) ) + b
    좌표 정규화: x_norm=(x-0.5)/0.5
    """
    def __init__(self, num_data: int, num_points: int, seed: int = 42,
                 grid_type: str = 'uniform', noise_std: float = 0.0,
                 subgroups: int = 3, adapt_params: dict | None = None):
        super().__init__()
        torch.manual_seed(seed)
        self.num_data   = num_data
        self.num_points = num_points
        self.seed       = seed
        self.grid_type  = grid_type
        self.is_train   = True
        self.noise_std  = float(noise_std)

        self.coord_scale  = 0.5
        self.coord_offset = 0.5

        def base_func(x: torch.Tensor) -> torch.Tensor:
            xx = torch.clamp(x, 0.0, 1.0)
            amp = torch.sqrt((xx * (1.0 - xx)).clamp_min(0.0))
            phase = (2.0 * math.pi * 1.05) / (xx + 0.05)
            return amp * torch.sin(phase)

        if grid_type == 'uniform':
            x_base = torch.linspace(0.0, 1.0, steps=self.num_points)
            meta = _subgroup_meta_from_grid(x_base, int(subgroups))
        elif grid_type == 'random':
            x_base = torch.rand(self.num_points).sort().values
            meta = _subgroup_meta_from_grid(x_base, int(subgroups))
        elif grid_type == 'adapted':
            ap = {} if adapt_params is None else dict(adapt_params)
            x_base, meta = build_adaptive_grid_1d(
                x_min=0.0, x_max=1.0, num_points=self.num_points,
                subgroups=int(subgroups),
                func=base_func,
                pilot_points=int(ap.get('pilot_points', 4096)),
                alpha=float(ap.get('alpha', 1.0)),
                smooth_sigma=float(ap.get('smooth_sigma', 5.0)),
                clip_quantile=float(ap.get('clip_quantile', 0.98)),
                min_per_group=int(ap.get('min_per_group', 1)),
                device=torch.device('cpu'),
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'.")

        self.subgroup_meta = meta
        self.x = x_base.unsqueeze(0).repeat(self.num_data, 1)

        torch.manual_seed(self.seed)
        b = (torch.randn(self.num_data, 1) * self.noise_std).repeat(1, self.num_points)

        def g(xx: torch.Tensor) -> torch.Tensor:
            amp = torch.sqrt((xx * (1.0 - xx)).clamp_min(0.0))
            phase = (2.0 * math.pi * 1.05) / (xx + 0.05)
            return amp * torch.sin(phase)

        y_raw = g(self.x) + b

        self.mean = y_raw.mean()
        self.std  = y_raw.std().clamp_min(1e-8)
        self.dataset = (y_raw - self.mean) / self.std

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int):
        if self.grid_type == 'random' and getattr(self, 'is_train', False):
            x_item = torch.rand(self.num_points).sort().values
            b = (torch.randn(1) * self.noise_std).repeat(self.num_points)
            amp   = torch.sqrt((x_item * (1.0 - x_item)).clamp_min(0.0))
            phase = (2.0 * math.pi * 1.05) / (x_item + 0.05)
            y_raw = amp * torch.sin(phase) + b
            y_norm = (y_raw - self.mean) / self.std
            return x_item.unsqueeze(-1), y_norm.unsqueeze(-1)

        return (
            self.x[idx].unsqueeze(-1),
            self.dataset[idx].unsqueeze(-1)
        )

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)




