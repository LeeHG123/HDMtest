import torch

class QuadraticDataset(torch.utils.data.Dataset):
    """
    y = a * x^2 + ε  (a ∈ {-1, +1}, ε ~ N(0,1)—구현상 함수별 상수 잡음)
    ─────────────────────────────────────────────
    저장 형태  : 전-데이터 Z-정규화된 값
    역변환 함수: inverse_transform(y_norm) → 원래 스케일

    좌표 정규화 규약(러너에서 사용):
      - 정의역 [-10,10] → [-1,1]로 맞추기 위해
        coord_scale = 10.0, coord_offset = 0.0  (x_norm = x / 10)
    """
    def __init__(self, num_data: int, num_points: int, seed: int = 42, grid_type: str = 'uniform'):
        super().__init__()
        torch.manual_seed(seed)

        self.num_data   = num_data
        self.num_points = num_points
        self.seed       = seed
        self.grid_type  = grid_type
        # Default: training dataset unless overwritten in get_dataset
        self.is_train   = True

        # 좌표 정규화 규약 명시
        self.coord_scale  = 10.0
        self.coord_offset = 0.0

        # 1) 기본 좌표 그리드 생성
        if grid_type == 'uniform':
            x_base = torch.linspace(start=-10., end=10., steps=self.num_points)
        elif grid_type == 'random':
            # 학습 시 __getitem__에서 항목별로 다시 샘플링하므로, 여기서는 한 번만 초기화
            x_base = (torch.rand(self.num_points) * 20 - 10).sort().values
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'. Choose 'uniform' or 'random'.")

        # (B, N)
        self.x = x_base.unsqueeze(0).repeat(self.num_data, 1)

        # 2) 전체 데이터 생성 (a, ε는 샘플별로)
        torch.manual_seed(self.seed)
        a   = (torch.randint(low=0, high=2, size=(self.num_data, 1)) * 2 - 1).repeat(1, self.num_points)  # {-1,+1}
        eps = torch.randn(self.num_data, 1).repeat(1, self.num_points)
        y   = a * (self.x ** 2) + eps  # (B, N)

        # 3) 전-데이터 Z-정규화 (μ, σ 저장)
        self.mean = y.mean()
        self.std  = y.std().clamp_min(1e-8)
        self.dataset = (y - self.mean) / self.std  # (B, N)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int):
        """
        반환:
          x: (N, 1) 좌표
          y: (N, 1) Z-정규화된 타깃
        """
        # 학습 중이고 grid_type=='random'이면 항목별로 좌표/함수를 재샘플
        if self.grid_type == 'random' and getattr(self, 'is_train', False):
            x_item = (torch.rand(self.num_points) * 20 - 10).sort().values  # (N,)
            a      = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
            eps    = torch.randn(1).repeat(self.num_points)
            y_item = a * (x_item ** 2) + eps
            y_item = (y_item - self.mean) / self.std
            return x_item.unsqueeze(-1), y_item.unsqueeze(-1)

        # 고정 그리드 샘플
        return (
            self.x[idx, :].unsqueeze(-1),          # (N, 1)
            self.dataset[idx, :].unsqueeze(-1)     # (N, 1)
        )

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Z-정규화된 값 → 원래 스케일로 되돌립니다.
        입력/출력 shape는 자유(…, N). device에 맞춰 브로드캐스트됩니다.
        """
        return y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)


