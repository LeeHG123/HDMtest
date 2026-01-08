# datasets/quadratic.py
import torch
import math

class QuadraticDataset(torch.utils.data.Dataset):
    """
    func_type ∈ {
        'quadratic', 'linear', 'circle', 'sin', 'doppler', 'sinc',
        'gaussian_bumps', 'am_sin'
    } 에 따라 1D 함수 데이터를 생성합니다.

      - quadratic: y = a * x^2 + ε,  a∈{-1,+1}
      - linear   : y = a * x   + ε,  a∈{-1,+1}
      - circle   : y = a * sqrt(r^2 - x^2) + ε,  a∈{-1,+1},  r∈{10,5} (정의역 [-10,10])
      - sin      : y = sin(x) + ε
      - sinc     : y = sinc(x) + ε, sinc(x)=sin(πx)/(πx)  (정의역 [-10,10])
      - doppler  : y = sqrt(x(1-x)) * sin((2π*1.05)/(x+0.05)) + b
                   (정의역 [0,1], b~N(0, noise_std^2) 는 "함수별 상수 오프셋")
      - gaussian_bumps:
            y = Σ_{i=1}^K A_i exp(-0.5 * ((x-c_i)/σ_i)^2) + ε
            (K=24 고정, {c_i,A_i,σ_i}는 seed로 고정된 단일 샘플 파라미터)
      - am_sin (amplitude-modulated sinusoid):
            envelope(x)= (1 + d_1 cos(ω_{m1} x + 0.3)) (1 + d_2 cos(ω_{m2} x - 1.1))
            taper(x)=0.55 + 0.45 exp(-0.5 (x/8)^2)
            y = taper(x) * envelope(x) * sin(ω_c x + φ) + ε
            (ω_c=2.3 고정, ω_{m1}=0.55, ω_{m2}=0.18, d_1=0.95, d_2=0.65, φ는 seed로 고정된 단일 샘플 위상)

    저장 형태  : 전-데이터 Z-정규화된 값 (mean/std는 데이터 전체에서 계산)
    역변환     : inverse_transform(y_norm) = y_norm * std + mean

    좌표 정규화(러너와 일치):
      - {quadratic, linear, sin, circle, sinc, gaussian_bumps, am_sin}:  x_norm = x / 10 ∈ [-1,1]
      - doppler:  x_norm = (x - 0.5) / 0.5 ∈ [-1,1]
    """
    def __init__(self, num_data: int, num_points: int, seed: int = 42,
                 grid_type: str = 'uniform', noise_std: float = 0.0,
                 func_type: str = 'quadratic'):
        super().__init__()
        torch.manual_seed(seed)

        func_type = str(func_type).lower()
        if func_type not in (
            'quadratic', 'linear', 'circle', 'sin', 'doppler', 'sinc',
            'gaussian_bumps', 'am_sin'
        ):
            raise ValueError(
                "func_type must be one of "
                "'quadratic','linear','circle','sin','doppler','sinc','gaussian_bumps','am_sin', "
                f"got {func_type}"
            )
        self.func_type  = func_type

        self.num_data   = num_data
        self.num_points = num_points
        self.seed       = seed
        self.grid_type  = grid_type
        self.is_train   = True
        self.noise_std  = float(noise_std)

        # 좌표/정의역 설정 (러너의 _coord_norm 과 일치)
        if self.func_type in ('doppler',):
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

        # Linear slope candidates: a ∈ {-1, 0, 1}
        self.linear_coeffs = torch.tensor([-1.0, 0.0, 1.0])

        # ─────────────────────────────────────────────────────────
        # (0) 단일 샘플 파라미터 고정 (gaussian_bumps / am_sin)
        #     전역 RNG를 가능한 덜 건드리기 위해 별도 Generator 사용
        # ─────────────────────────────────────────────────────────
        if self.func_type == 'gaussian_bumps':
            g = torch.Generator(device='cpu').manual_seed(self.seed + 1701)
            self.gb_num_bumps = 24

            # 고주파 성분 완화(요청 반영): σ 를 전역 스케일링하여 bump 폭을 확장
            #   - 원래 σ: log-uniform in [0.10, 0.60]
            #   - 적용 σ: σ_scaled = gb_sigma_scale * σ
            self.gb_sigma_scale = 3.0

            span = (self._xmax - 0.7) - (self._xmin + 0.7)
            self.gb_centers = (self._xmin + 0.7) + torch.rand(self.gb_num_bumps, generator=g) * span
            self.gb_amps = torch.randn(self.gb_num_bumps, generator=g)

            # sigmas (pre-scale): log-uniform in [0.10, 0.60]
            sigma_min, sigma_max = 0.10, 0.60
            u = torch.rand(self.gb_num_bumps, generator=g)
            log_min = math.log10(sigma_min)
            log_max = math.log10(sigma_max)
            self.gb_sigmas = torch.pow(10.0, log_min + (log_max - log_min) * u) * self.gb_sigma_scale

        if self.func_type == 'am_sin':
            g = torch.Generator(device='cpu').manual_seed(self.seed + 2909)
            self.am_carrier_w = 2.3
            # 더 다양한 진폭 변조(envelope)를 위해 저주파 성분을 하나 더 추가.
            # carrier 주파수(= am_carrier_w)는 그대로 유지한다.
            self.am_mod_w = 0.55      # ω_{m1}
            self.am_mod_w2 = 0.18     # ω_{m2} (더 저주파)
            self.am_mod_depth = 0.95  # d_1
            self.am_mod_depth2 = 0.65 # d_2
            self.am_phase = float(2.0 * math.pi * torch.rand((), generator=g))

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

        else:
            # 공통: 함수별 상수 잡음 ε
            eps = torch.randn(self.num_data, 1) * self.noise_std
            eps = eps.repeat(1, self.num_points)

            if self.func_type == 'quadratic':
                a = (torch.randint(low=0, high=2, size=(self.num_data, 1)) * 2 - 1).repeat(1, self.num_points)
                y = a * (self.x ** 2) + eps

            elif self.func_type == 'linear':
                a_choices = self.linear_coeffs.to(self.x.device).type_as(self.x)
                a_idx = torch.randint(low=0, high=a_choices.numel(), size=(self.num_data, 1), device=self.x.device)
                a = a_choices[a_idx].repeat(1, self.num_points)
                y = a * self.x + eps

            elif self.func_type == 'sin':
                y = torch.sin(self.x) + eps

            elif self.func_type == 'sinc':
                y = torch.sinc(self.x) + eps

            elif self.func_type == 'gaussian_bumps':
                # 단일 샘플 base function을 (1,N)에서 만들고 (B,N)로 반복
                x0 = self.x[0:1, :]  # (1,N)
                centers = self.gb_centers.view(1, 1, -1)  # (1,1,K)
                amps    = self.gb_amps.view(1, 1, -1)     # (1,1,K)
                sigmas  = self.gb_sigmas.view(1, 1, -1)   # (1,1,K)

                diff = (x0.unsqueeze(-1) - centers) / sigmas     # (1,N,K)
                y0   = (amps * torch.exp(-0.5 * diff.pow(2))).sum(dim=-1)  # (1,N)
                y    = y0.repeat(self.num_data, 1) + eps

            elif self.func_type == 'am_sin':
                x0 = self.x[0:1, :]  # (1,N)
                envelope = (
                    1.0 + self.am_mod_depth * torch.cos(self.am_mod_w * x0 + 0.3)
                ) * (
                    1.0 + self.am_mod_depth2 * torch.cos(self.am_mod_w2 * x0 - 1.1)
                )
                y0 = envelope * torch.sin(self.am_carrier_w * x0 + self.am_phase)
                taper = torch.exp(-0.5 * (x0 / 8.0).pow(2)) * 0.45 + 0.55
                y0 = y0 * taper
                y = y0.repeat(self.num_data, 1) + eps

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

            else:
                eps = torch.randn(1).repeat(self.num_points) * self.noise_std

                if self.func_type == 'quadratic':
                    a = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
                    y_item = a * (x_item ** 2) + eps

                elif self.func_type == 'linear':
                    a_choices = self.linear_coeffs.to(x_item.device).type_as(x_item)
                    a = a_choices[torch.randint(low=0, high=a_choices.numel(), size=(1,), device=x_item.device)].item()
                    y_item = a * x_item + eps

                elif self.func_type == 'sin':
                    y_item = torch.sin(x_item) + eps

                elif self.func_type == 'sinc':
                    y_item = torch.sinc(x_item) + eps

                elif self.func_type == 'gaussian_bumps':
                    dev = x_item.device
                    centers = self.gb_centers.to(dev)  # (K,)
                    amps    = self.gb_amps.to(dev)     # (K,)
                    sigmas  = self.gb_sigmas.to(dev)   # (K,)

                    diff = (x_item.unsqueeze(-1) - centers.view(1, -1)) / sigmas.view(1, -1)  # (N,K)
                    y0 = (amps.view(1, -1) * torch.exp(-0.5 * diff.pow(2))).sum(dim=-1)       # (N,)
                    y_item = y0 + eps

                elif self.func_type == 'am_sin':
                    envelope = (
                        1.0 + self.am_mod_depth * torch.cos(self.am_mod_w * x_item + 0.3)
                    ) * (
                        1.0 + self.am_mod_depth2 * torch.cos(self.am_mod_w2 * x_item - 1.1)
                    )
                    y0 = envelope * torch.sin(self.am_carrier_w * x_item + self.am_phase)
                    taper = torch.exp(-0.5 * (x_item / 8.0).pow(2)) * 0.45 + 0.55
                    y_item = (y0 * taper) + eps

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

        # 공통(비 Doppler)
        eps = torch.randn(B, 1, device=dev) * self.noise_std
        eps = eps.repeat(1, N)

        if self.func_type == 'quadratic':
            a   = (torch.randint(low=0, high=2, size=(B, 1), device=dev) * 2 - 1).repeat(1, N)
            y_raw = a * (x_batch ** 2) + eps
            return y_raw

        if self.func_type == 'linear':
            a_choices = self.linear_coeffs.to(dev).type_as(x_batch)
            a_idx = torch.randint(low=0, high=a_choices.numel(), size=(B, 1), device=dev)
            a = a_choices[a_idx].repeat(1, N)
            y_raw = a * x_batch + eps
            return y_raw

        if self.func_type == 'sin':
            return torch.sin(x_batch) + eps

        if self.func_type == 'sinc':
            return torch.sinc(x_batch) + eps

        if self.func_type == 'gaussian_bumps':
            centers = self.gb_centers.to(dev).view(1, 1, -1)  # (1,1,K)
            amps    = self.gb_amps.to(dev).view(1, 1, -1)
            sigmas  = self.gb_sigmas.to(dev).view(1, 1, -1)

            diff = (x_batch.unsqueeze(-1) - centers) / sigmas         # (B,N,K)
            y0   = (amps * torch.exp(-0.5 * diff.pow(2))).sum(dim=-1) # (B,N)
            return y0 + eps

        if self.func_type == 'am_sin':
            envelope = (
                1.0 + self.am_mod_depth * torch.cos(self.am_mod_w * x_batch + 0.3)
            ) * (
                1.0 + self.am_mod_depth2 * torch.cos(self.am_mod_w2 * x_batch - 1.1)
            )
            y0 = envelope * torch.sin(self.am_carrier_w * x_batch + self.am_phase)
            taper = torch.exp(-0.5 * (x_batch / 8.0).pow(2)) * 0.45 + 0.55
            y0 = y0 * taper
            return y0 + eps

        # circle
        a = (torch.randint(low=0, high=2, size=(B, 1), device=dev) * 2 - 1).repeat(1, N)
        r_choices = self.circle_radii.to(dev)
        r_idx = torch.randint(low=0, high=r_choices.numel(), size=(B, 1), device=dev)
        r = r_choices[r_idx]  # (B,1)
        y_raw = a * torch.sqrt((r.pow(2) - x_batch.pow(2)).clamp_min(0.0)) + eps
        return y_raw
















