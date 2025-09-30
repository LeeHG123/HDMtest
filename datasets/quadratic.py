import torch

class QuadraticDataset(torch.utils.data.Dataset):
    """
    y = a * x^2 + ε  (a ∈ {-1, +1}, ε ~ N(0,1)—구현상 함수별 상수 잡음)
    저장 형태  : 전-데이터 Z-정규화된 값
    역변환     : inverse_transform(y_norm)
    좌표 정규화:
      coord_scale=10.0, coord_offset=0.0  (x_norm = x/10)
    """
    def __init__(self, num_data: int, num_points: int, seed: int = 42,
                 grid_type: str = 'uniform', noise_std: float = 0.0):
        super().__init__()
        torch.manual_seed(seed)

        self.num_data   = num_data
        self.num_points = num_points
        self.seed       = seed
        self.grid_type  = grid_type
        self.is_train   = True
        self.noise_std  = float(noise_std)        

        self.coord_scale  = 10.0
        self.coord_offset = 0.0

        def base_func(x: torch.Tensor) -> torch.Tensor:
            return x**2

        # 1) 그리드
        if grid_type == 'uniform':
            x_base = torch.linspace(start=-10., end=10., steps=self.num_points)
        elif grid_type == 'random':
            x_base = (torch.rand(self.num_points) * 20 - 10).sort().values         
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'. Choose 'uniform' or 'random'")

        self.x = x_base.unsqueeze(0).repeat(self.num_data, 1)

        # 2) 데이터 생성
        torch.manual_seed(self.seed)
        a   = (torch.randint(low=0, high=2, size=(self.num_data, 1)) * 2 - 1).repeat(1, self.num_points)
        eps = torch.randn(self.num_data, 1).repeat(1, self.num_points) * self.noise_std
        y   = a * (self.x ** 2) + eps

        # 3) Z-정규화
        self.mean = y.mean()
        self.std  = y.std().clamp_min(1e-8)
        self.dataset = (y - self.mean) / self.std

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int):
        if self.grid_type == 'random' and getattr(self, 'is_train', False):
            x_item = (torch.rand(self.num_points) * 20 - 10).sort().values
            a      = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
            eps    = torch.randn(1).repeat(self.num_points) * self.noise_std
            y_item = a * (x_item ** 2) + eps
            y_item = (y_item - self.mean) / self.std
            return x_item.unsqueeze(-1), y_item.unsqueeze(-1)      

        return (
            self.x[idx, :].unsqueeze(-1),
            self.dataset[idx, :].unsqueeze(-1)
        )

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)




