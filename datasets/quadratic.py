# datasets/quadratic.py
import torch
import math

class QuadraticDataset(torch.utils.data.Dataset):
    """
    func_type ∈ {'quadratic', 'linear', 'circle', 'sin', 'doppler', 'sinc', 'rq', 'matern', 'matern12', 'matern32', 'matern52', 'blocks'} 에 따라
      - quadratic: y = a * x^2 + ε,  a∈{-1,+1}
      - linear   : y = a * x   + ε,  a∈{-1,+1}
      - circle   : y = a * sqrt(r^2 - x^2) + ε,  a∈{-1,+1},  r∈{10,5} (정의역 [-10,10])
      - sin      : y = sin(x) + ε
      - doppler  : y = sqrt(x(1-x)) * sin((2π*1.05)/(x+0.05)) + b
                   (정의역 [0,1], a를 곱하지 않음, b~N(0, noise_std^2) 는 "함수별 상수 오프셋")
      - sinc     : y = sinc(x) + ε, sinc(x) = sin(πx)/(πx)  (정의역 [-10,10], a 미사용)
      - rq       : y = Σ_j c_j k_RQ(|x-ξ_j|; α_j, ℓ_j) + ε
      - matern{12,32,52} / matern(=32 기본):
             y = Σ_j c_j k_Matern_ν(|x-ξ_j|; ℓ_j) + ε
      - blocks   : y = Σ_{k=1}^{11} h_k 1_{[b_k,1]}(x) + ε  (정의역 [0,1], a 미사용)   # ← ADD
        (모두 정의역 [-10,10]이 기본이나, doppler/blocks는 [0,1])
    저장 형태  : 전-데이터 Z-정규화된 값 (mean/std는 데이터 전체에서 계산)
    역변환     : inverse_transform(y_norm) = y_norm * std + mean

    좌표 정규화(러너와 일치):
      - {quadratic, linear, sin, circle, sinc, rq, matern, matern12, matern32, matern52}:  x_norm = x / 10 ∈ [-1,1]
      - doppler / blocks:  x_norm = (x - 0.5) / 0.5 ∈ [-1,1]                               # ← UPDATE
    """
    def __init__(self, num_data: int, num_points: int, seed: int = 42,
                 grid_type: str = 'uniform', noise_std: float = 0.0,
                 func_type: str = 'quadratic'):
        super().__init__()
        torch.manual_seed(seed)

        func_type = str(func_type).lower()
        if func_type not in ('quadratic', 'linear', 'circle', 'sin', 'doppler', 'sinc', 'rq', 'matern', 'matern12', 'matern32', 'matern52', 'blocks', 'gaussmix','gaussian','gmm'):
            raise ValueError(
                f"func_type must be one of "
                f"'quadratic','linear','circle','sin','doppler','sinc','rq','matern','matern12','matern32','matern52','blocks', 'gaussmix','gaussian','gmm', got {func_type}"
            )
        self.func_type  = func_type

        self.num_data   = num_data
        self.num_points = num_points
        self.seed       = seed
        self.grid_type  = grid_type
        self.is_train   = True
        self.noise_std  = float(noise_std)

        # 좌표/정의역 설정 (러너의 _coord_norm 과 일치)
        if self.func_type in ('doppler', 'blocks'): 
            # [0,1] → [-1,1]
            self.coord_scale  = 0.5
            self.coord_offset = 0.5
            self._xmin, self._xmax = 0.0, 1.0
        else:
            # [-10,10] → [-1,1]
            self.coord_scale  = 10.0
            self.coord_offset = 0.0
            self._xmin, self._xmax = -10.0, 10.0

        # Circle 반지름(도메인 [-10,10]과 일치)
        self.radius = 10.0
        self.circle_radii = torch.tensor([10.0, 5.0])

        # 1) 좌표 그리드 생성
        if grid_type == 'uniform':
            x_base = torch.linspace(start=self._xmin, end=self._xmax, steps=self.num_points)
        elif grid_type == 'random':
            x_base = (torch.rand(self.num_points) * (self._xmax - self._xmin) + self._xmin).sort().values
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'. Choose 'uniform' or 'random'")

        self.x = x_base.unsqueeze(0).repeat(self.num_data, 1)  # (B, N)

        # 2) 데이터 생성 (ε 또는 b: 함수별 상수 잡음)
        torch.manual_seed(self.seed)

        if self.func_type == 'doppler':
            # b: (B,1) → (B,N)
            b = torch.randn(self.num_data, 1) * self.noise_std
            b = b.repeat(1, self.num_points)

            xx = self.x.clamp(0.0, 1.0)
            amp   = torch.sqrt((xx * (1.0 - xx)).clamp_min(0.0))
            phase = (2.0 * math.pi * 1.05) / (xx + 0.05)
            y = amp * torch.sin(phase) + b

        elif self.func_type in ('rq', 'matern', 'matern12', 'matern32', 'matern52'):
            y = self._kernel_mixture(self.x, kind=self.func_type)

        elif self.func_type == 'blocks':                                   
            eps = torch.randn(self.num_data, 1) * self.noise_std
            eps = eps.repeat(1, self.num_points)
            y = self._blocks_batch(self.x) + eps            

        else:
            # 공통: 함수별 상수 잡음 ε
            eps = torch.randn(self.num_data, 1) * self.noise_std
            eps = eps.repeat(1, self.num_points)

            if self.func_type == 'quadratic':
                a = (torch.randint(low=0, high=2, size=(self.num_data, 1)) * 2 - 1).repeat(1, self.num_points)
                y = a * (self.x ** 2) + eps
            elif self.func_type == 'linear':
                a = (torch.randint(low=0, high=2, size=(self.num_data, 1)) * 2 - 1).repeat(1, self.num_points)
                y = a * (self.x) + eps
            elif self.func_type == 'sin':
                y = torch.sin(self.x) + eps
            elif self.func_type == 'sinc':
                # torch.sinc(u) = sin(πu)/(πu)  
                y = torch.sinc(self.x) + eps     
            elif self.func_type in ('gaussmix', 'gaussian', 'gmm'):  
                eps = torch.randn(self.num_data, 1) * self.noise_std
                eps = eps.repeat(1, self.num_points)
                y = self._gaussmix_batch(self.x) + eps                           
            else:  # 'circle'
                a = (torch.randint(low=0, high=2, size=(self.num_data, 1)) * 2 - 1).repeat(1, self.num_points)
                r_choices = self.circle_radii.to(self.x.device)
                r_idx = torch.randint(low=0, high=r_choices.numel(), size=(self.num_data, 1), device=self.x.device)
                r = r_choices[r_idx]  # (B,1)
                y = a * torch.sqrt((r.pow(2) - self.x.pow(2)).clamp_min(0.0)) + eps

        # 3) Z-정규화 통계 및 저장
        self.mean = y.mean()
        self.std  = y.std().clamp_min(1e-8)
        self.dataset = (y - self.mean) / self.std

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int):
        """
        grid_type=='random' & train: 매 호출마다 새 좌표/함수 샘플 (해상도-무관 학습)
        """
        if self.grid_type == 'random' and getattr(self, 'is_train', False):
            x_item = (torch.rand(self.num_points) * (self._xmax - self._xmin) + self._xmin).sort().values

            if self.func_type == 'doppler':
                b = (torch.randn(1) * self.noise_std).repeat(self.num_points)
                xx = x_item.clamp(0.0, 1.0)
                amp   = torch.sqrt((xx * (1.0 - xx)).clamp_min(0.0))
                phase = (2.0 * math.pi * 1.05) / (xx + 0.05)
                y_item = amp * torch.sin(phase) + b

            elif self.func_type in ('rq', 'matern', 'matern12', 'matern32', 'matern52'):
                y_item = self._kernel_mixture(x_item.unsqueeze(0), kind=self.func_type)[0]

            elif self.func_type == 'blocks':                                  
                eps = torch.randn(1).repeat(self.num_points) * self.noise_std
                y_item = self._blocks_batch(x_item.unsqueeze(0))[0] + eps

            else:
                eps = torch.randn(1).repeat(self.num_points) * self.noise_std
                if self.func_type == 'quadratic':
                    a = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
                    y_item = a * (x_item ** 2) + eps
                elif self.func_type == 'linear':
                    a = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
                    y_item = a * (x_item) + eps
                elif self.func_type == 'sin':
                    y_item = torch.sin(x_item) + eps
                elif self.func_type == 'sinc':
                    y_item = torch.sinc(x_item) + eps
                elif self.func_type in ('gaussmix', 'gaussian', 'gmm'):
                    eps = torch.randn(1).repeat(self.num_points) * self.noise_std
                    y_item = self._gaussmix_batch(x_item.unsqueeze(0))[0] + eps                    
                else:  # 'circle'
                    a = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
                    r_choices = self.circle_radii.to(x_item.device)
                    r = r_choices[torch.randint(low=0, high=r_choices.numel(), size=(1,))].item()
                    r2 = r * r
                    y_item = a * torch.sqrt((r2 - x_item.pow(2)).clamp_min(0.0)) + eps

            y_item = (y_item - self.mean) / self.std
            return x_item.unsqueeze(-1), y_item.unsqueeze(-1)

        # 고정 그리드 샘플
        return (
            self.x[idx, :].unsqueeze(-1),
            self.dataset[idx, :].unsqueeze(-1)
        )

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)

    # ── Gaussian mixture 유틸리티 ─────────────────────────────────────────
    def _gm_params(self, device=None, dtype=torch.float32):
        """
        고정 혼합 파라미터(계수/평균/표준편차)를 반환.
        - 정의역: [-10, 10]
        - 다양한 대역폭을 위해 σ는 매우 작음(≈0.35)부터 큼(≈3.0)까지 혼합
        """
        dev = device if device is not None else getattr(self.x, "device", torch.device("cpu"))
        dt  = dtype
        # 평균(μ): 도메인 전역에 고르게 분포
        mu = torch.tensor([-8.5, -6.0, -3.5, -1.5, 0.0, 1.2, 3.0, 5.5, 7.5, 9.0], device=dev, dtype=dt)
        # 표준편차(σ): 다중 스케일(좁은 피크 ~ 넓은 범프)
        sigma = torch.tensor([0.35, 0.50, 0.80, 1.20, 0.60, 2.50, 1.80, 0.90, 3.00, 0.45], device=dev, dtype=dt)
        # 계수(c): 양/음 혼합(상쇄·강조 패턴 유도)
        coef = torch.tensor([ 1.35, -0.90,  1.10, -1.70,  0.60,
                            2.00, -1.30,  1.50, -0.85,  1.20], device=dev, dtype=dt)
        return coef, mu, sigma

    def _gaussmix_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        x_batch: (B,N) 또는 (N,)
        y = Σ_j c_j * exp( -0.5 * ((x-μ_j)/σ_j)^2 )
        """
        if x_batch.dim() == 1:
            x_batch = x_batch.unsqueeze(0)  # (1,N)
        B, N = x_batch.shape
        dev, dt = x_batch.device, x_batch.dtype
        c, mu, sigma = self._gm_params(device=dev, dtype=dt)  # (M,)

        X = x_batch.unsqueeze(-1)           # (B,N,1)
        mu = mu.view(1, 1, -1)              # (1,1,M)
        sg = sigma.view(1, 1, -1)           # (1,1,M)
        g  = torch.exp(-0.5 * ((X - mu) / sg).pow(2))     # (B,N,M)
        y  = torch.matmul(g, c.view(-1, 1)).squeeze(-1)   # (B,N)
        return y


    # ── Blocks 유틸리티 ────────────────────────────────────────────────
    @staticmethod
    def _blocks_params(device=None, dtype=torch.float32):                 # ← ADD
        b = torch.tensor([0.10, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81],
                         device=device, dtype=dtype)
        h = torch.tensor([ 4.0, -5.0,  3.0, -4.0,  5.0, -4.0,  4.0, -5.0,  4.0, -4.0,  4.0],
                         device=device, dtype=dtype)
        return b, h

    def _blocks_batch(self, x_batch: torch.Tensor) -> torch.Tensor:       # ← ADD
        """
        x_batch: (B, N) or (N,) in [0,1]
        return : (B, N) with g(x) = Σ_k h_k 1_{[b_k,1]}(x) = Σ_k h_k * 1(x≥b_k)
        """
        if x_batch.dim() == 1:
            x_batch = x_batch.unsqueeze(0)
        B, N = x_batch.shape
        dev, dt = x_batch.device, x_batch.dtype
        b, h = self._blocks_params(device=dev, dtype=dt)       # (11,), (11,)
        X = x_batch.unsqueeze(-1)                               # (B,N,1)
        step = (X >= b.view(1, 1, -1)).to(dt)                  # (B,N,11)
        y = torch.matmul(step, h.view(-1, 1)).squeeze(-1)      # (B,N)
        return y        

    # ── 커널 폐형식 (1D) ─────────────────────────────────────────────
    @staticmethod
    def _rq_kernel(r: torch.Tensor, ell: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # r, ell, alpha: 브로드캐스트 호환 (…, N, M)
        base = 1.0 + (r * r) / (2.0 * alpha * (ell * ell))
        return torch.pow(base, -alpha)

    @staticmethod
    def _matern12(r: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        return torch.exp(-r / ell)

    @staticmethod
    def _matern32(r: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        z = (math.sqrt(3.0) * r) / ell
        return (1.0 + z) * torch.exp(-z)

    @staticmethod
    def _matern52(r: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        z = (math.sqrt(5.0) * r) / ell
        return (1.0 + z + (z * z) / 3.0) * torch.exp(-z)

    def _kernel_mixture(self, x_batch: torch.Tensor, kind: str) -> torch.Tensor:
        """
        x_batch: (B, N)  in [-10,10]
        return : (B, N)  y_raw = Σ_j c_j k(|x-ξ_j|; θ_j) + (상수 잡음)
        """
        dev = x_batch.device
        B, N = x_batch.shape
        M = 5  # mixture 개수(필요시 조절)

        # 중심/길이척도/계수/형상 매개변수 샘플링(도메인 [-10,10])
        xi   = (torch.rand(B, M, device=dev) * (self._xmax - self._xmin) + self._xmin)  # (B,M)
        ell  = (torch.rand(B, M, device=dev) * (3.0 - 0.5) + 0.5)                       # ℓ ∈ [0.5,3.0]
        coef = torch.randn(B, M, device=dev)                                            # c_j ~ N(0,1)

        x_exp = x_batch.unsqueeze(-1)   # (B,N,1)
        r = torch.abs(x_exp - xi.unsqueeze(1))  # (B,N,M)
        ell_b = ell.unsqueeze(1)                 # (B,1,M)

        if kind in ('matern', 'matern32'):
            k = self._matern32(r, ell_b)
        elif kind == 'matern12':
            k = self._matern12(r, ell_b)
        elif kind == 'matern52':
            k = self._matern52(r, ell_b)
        elif kind == 'rq':
            alpha = (torch.rand(B, M, device=dev) * (3.0 - 0.5) + 0.5)     # α ∈ [0.5,3.0]
            alpha_b = alpha.unsqueeze(1)                                    # (B,1,M)
            k = self._rq_kernel(r, ell_b, alpha_b)
        else:
            raise ValueError(f"unknown kernel-mixture kind: {kind}")

        # Σ_j c_j k(|x-ξ_j|; θ_j)
        y = (k * coef.unsqueeze(1)).sum(dim=2)  # (B,N)

        # (기존 규칙과 동일하게) 상수 잡음만 추가
        if self.noise_std > 0:
            eps = torch.randn(B, 1, device=dev) * self.noise_std
            y = y + eps.repeat(1, N)
        return y

    # ─────────────────────────────────────────────────────────
    # 해상도-자유 검증/샘플링용 원스케일 함수 생성
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_raw(self, x_batch: torch.Tensor, *, device=None) -> torch.Tensor:
        """
        x_batch: (B, N) — 원 좌표계 값
          - doppler: x ∈ [0,1]
          - 기타   : x ∈ [-10,10]
        return:  (B, N) — y_raw (정규화 전)
        """
        dev = device or x_batch.device
        x_batch = x_batch.to(dev)
        B, N = x_batch.shape

        if self.func_type == 'doppler':
            b = torch.randn(B, 1, device=dev) * self.noise_std
            b = b.repeat(1, N)
            xx = torch.clamp(x_batch, 0.0, 1.0)
            amp   = torch.sqrt((xx * (1.0 - xx)).clamp_min(0.0))
            phase = (2.0 * math.pi * 1.05) / (xx + 0.05)
            y_raw = amp * torch.sin(phase) + b
            return y_raw

        if self.func_type in ('rq', 'matern', 'matern12', 'matern32', 'matern52'):
            return self._kernel_mixture(x_batch, kind=self.func_type)        

        if self.func_type == 'blocks':                                  
            eps = torch.randn(B, 1, device=dev) * self.noise_std
            eps = eps.repeat(1, N)
            return self._blocks_batch(x_batch) + eps  

        if self.func_type in ('gaussmix', 'gaussian', 'gmm'):  
            eps = torch.randn(B, 1, device=dev) * self.noise_std
            eps = eps.repeat(1, N)
            return self._gaussmix_batch(x_batch) + eps

        # 공통(비 Doppler)
        eps = torch.randn(B, 1, device=dev) * self.noise_std
        eps = eps.repeat(1, N)

        if self.func_type == 'quadratic':
            a   = (torch.randint(low=0, high=2, size=(B, 1), device=dev) * 2 - 1).repeat(1, N)
            y_raw = a * (x_batch ** 2) + eps
        elif self.func_type == 'linear':
            a   = (torch.randint(low=0, high=2, size=(B, 1), device=dev) * 2 - 1).repeat(1, N)
            y_raw = a * (x_batch) + eps
        elif self.func_type == 'sin':
            y_raw = torch.sin(x_batch) + eps
        elif self.func_type == 'sinc':
            y_raw = torch.sinc(x_batch) + eps             
        else:  # 'circle'
            a = (torch.randint(low=0, high=2, size=(B, 1), device=dev) * 2 - 1).repeat(1, N)
            r_choices = self.circle_radii.to(dev)
            r_idx = torch.randint(low=0, high=r_choices.numel(), size=(B, 1), device=dev)
            r = r_choices[r_idx]  # (B,1)
            y_raw = a * torch.sqrt((r.pow(2) - x_batch.pow(2)).clamp_min(0.0)) + eps
            return y_raw

        return y_raw









