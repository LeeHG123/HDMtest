import torch
import math

class GaussianDataset(torch.utils.data.Dataset):
    """
    y = log φ(x) + ε  (φ: 표준정규 PDF)
    ─────────────────────────────────────
    저장 형태  : z-score 정규화된 log-pdf
    역변환 함수: self.inverse_transform(...)
    """

    def __init__(self, num_data: int, num_points: int, seed: int = 42, grid_type: str = 'uniform'):
        super().__init__()
        torch.manual_seed(seed)

        if grid_type == 'uniform':
            x_base = torch.linspace(-10., 10., steps=num_points)
        elif grid_type == 'random':
            x_base = (torch.rand(num_points) * 20 - 10).sort().values
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'. Choose 'uniform' or 'random'.")
            
        self.x = x_base.unsqueeze(0).repeat(num_data, 1)                  # (B,N)

        # 2) log-pdf (= –0.5x² – log√(2π))
        log_phi = -0.5 * self.x**2 - 0.5 * math.log(2*math.pi)            # (B,N)

        # 3) ε ~ 𝒩(0, 10⁻³) 추가 (log 공간에서도 충분히 작음)
        eps = torch.randn_like(log_phi) * 1e-2
        log_phi_noisy = log_phi + eps

        # 4) 전-데이터 Z-정규화 (μ,σ 저장)
        self.mean = log_phi_noisy.mean()
        self.std  = log_phi_noisy.std()
        self.dataset = (log_phi_noisy - self.mean) / self.std             # (B,N)

    # ── 필수 메서드 ────────────────────────────────────────────────
    def __len__(self): return self.x.size(0)

    def __getitem__(self, idx):
        return (
            self.x[idx].unsqueeze(-1),                # (N,1)
            self.dataset[idx].unsqueeze(-1)           # (N,1)
        )

    # ── 역변환 : 네트워크 출력 → 원래 φ(x) 스케일 ──────────────────
    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        입력  : z-score 정규화된 log-pdf   (…,N)
        반환값: 원래 φ(x) 값              (…,N)
        """
        y_log = y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)
        return torch.exp(y_log)
