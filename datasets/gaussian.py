# datasets/gaussian.py
import torch
import math

class GaussianDataset(torch.utils.data.Dataset):
    """
    y = log φ(x) + ε 를 z-score 정규화하여 저장
    역변환: exp(y_log)
    """
    def __init__(self, num_data: int, num_points: int, seed: int = 42,
                 grid_type: str = 'uniform', 
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
        elif grid_type == 'random':
            x_base = (torch.rand(num_points) * 20 - 10).sort().values
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'.")

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

        # 고정 그리드 샘플
        return (
            self.x[idx].unsqueeze(-1),
            self.dataset[idx].unsqueeze(-1)
        )

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        y_log = y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)
        return torch.exp(y_log)




