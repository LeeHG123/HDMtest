import torch
import math

class DopplerDataset(torch.utils.data.Dataset):
    """
    g(x) = sqrt(x(1-x)) * sin( (2π*1.05)/(x+0.05) ) + b
      - x ∈ (0,1) (여기서는 [0,1] 균등격자 또는 (0,1) 랜덤 샘플)
      - b ~ N(0,1): 각 '함수(샘플)'마다 한 번만 뽑아 모든 좌표에 동일하게 더함 (함수별 상수 잡음)
    저장 형태 : 학습셋 전역( B×N ) Z-정규화
    역변환    : inverse_transform(y_norm) → 원스케일(g+b)
    API·반환  : QuadraticDataset과 동일 시그니처/형태

    좌표 정규화 규약(러너에서 사용):
      - [-1, 1]로의 선형 정규화를 위해
        coord_scale = 0.5, coord_offset = 0.5  (x_norm = (x - 0.5) / 0.5 = 2x - 1)
    """
    def __init__(self, num_data: int, num_points: int, seed: int = 42, grid_type: str = 'uniform', noise_std: float = 0.01):
        super().__init__()
        torch.manual_seed(seed)
        self.num_data   = num_data
        self.num_points = num_points
        self.seed       = seed
        self.grid_type  = grid_type
        self.is_train   = True  # get_dataset에서 덮어씀
        self.noise_std  = float(noise_std)

        # 좌표 스케일(좌표 채널 정규화용)
        # Doppler는 정의역 (0,1) → [-1,1]로 맞추기 위해 (x - 0.5)/0.5 사용
        self.coord_scale  = 0.5
        self.coord_offset = 0.5

        # 1) 기본 좌표 그리드
        if grid_type == 'uniform':
            x_base = torch.linspace(0.0, 1.0, steps=self.num_points)  # [0,1]
        elif grid_type == 'random':
            # 학습 중 __getitem__에서 항목별 재샘플이 이뤄지므로, 여기서는 한 번만 초기화
            x_base = torch.rand(self.num_points).sort().values  # (0,1) 정렬
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}' (use 'uniform' or 'random').")

        # (B, N)
        self.x = x_base.unsqueeze(0).repeat(self.num_data, 1)

        # 2) 전체 데이터 생성: 함수별 상수 잡음 b ~ N(0,noise_std^2)을 각 샘플에 1개만
        torch.manual_seed(self.seed)
        # b: (B, 1) → (B, N)
        b = (torch.randn(self.num_data, 1) * self.noise_std).repeat(1, self.num_points)

        # g(x): (B, N)
        def g(xx: torch.Tensor) -> torch.Tensor:
            # 안전한 sqrt를 위해 clamp_min(0.)
            amp = torch.sqrt((xx * (1.0 - xx)).clamp_min(0.0))
            phase = (2.0 * math.pi * 1.05) / (xx + 0.05)
            return amp * torch.sin(phase)

        y_raw = g(self.x) + b  # (B, N)

        # 3) 전-데이터 Z-정규화 (μ, σ 저장)
        self.mean = y_raw.mean()
        self.std  = y_raw.std().clamp_min(1e-8)
        self.dataset = (y_raw - self.mean) / self.std  # (B, N)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int):
        """
        반환: (x: (N,1), y_norm: (N,1))  — 항상 학습 통계(μ_train, σ_train) 기준
        grid_type=='random' & 학습 중이면 좌표/함수를 항목별로 재샘플하여 계약 유지
        """
        if self.grid_type == 'random' and getattr(self, 'is_train', False):
            x_item = torch.rand(self.num_points).sort().values  # (0,1) 정렬
            # 함수별 상수 잡음 b 한 번만
            b = (torch.randn(1) * self.noise_std).repeat(self.num_points)
            # g(x)
            amp   = torch.sqrt((x_item * (1.0 - x_item)).clamp_min(0.0))
            phase = (2.0 * math.pi * 1.05) / (x_item + 0.05)
            y_raw = amp * torch.sin(phase) + b
            y_norm = (y_raw - self.mean) / self.std
            return x_item.unsqueeze(-1), y_norm.unsqueeze(-1)

        return (
            self.x[idx].unsqueeze(-1),           # (N,1)
            self.dataset[idx].unsqueeze(-1)      # (N,1)
        )

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Z-정규화된 값을 원래 스케일로 복원 (…,N) → (…,N)"""
        return y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)

