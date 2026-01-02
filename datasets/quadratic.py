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
      - linear   : y = a * x   + ε,  a∈{-2,-1/2,-1,1/2,1,2}
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
        self.num_data   = int(num_data)
        self.num_points = int(num_points)
        self.grid_type  = str(grid_type).lower()
        self.noise_std  = float(noise_std)

        if self.grid_type not in ('uniform', 'random'):
            raise ValueError("grid_type must be 'uniform' or 'random'")

        # doppler만 [0,1], 나머지는 [-10,10]
        if self.func_type == 'doppler':
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

        # Linear 기울기 후보
        #  a ∈ {-2, -1/2, -1, 1/2, 1, 2}
        self.linear_coeffs = torch.tensor([-2.0, -0.5, -1.0, 0.5, 1.0, 2.0])

        # ─────────────────────────────────────────────────────────
        # (0) 단일 샘플 파라미터 고정 (gaussian_bumps / am_sin)
        #     전역 RNG를 가능한 덜 건드리기 위해 별도 Generator 사용
        # ─────────────────────────────────────────────────────────
        g = torch.Generator()
        g.manual_seed(seed + 12345)

        # gaussian_bumps 파라미터(단일 샘플)
        K = 24
        centers = torch.linspace(-8.0, 8.0, K)  # 고정 center
        # amplitude/width는 seed로 고정
        amps = 0.75 + 0.5 * torch.rand(K, generator=g)
        sigmas = 0.35 + 0.15 * torch.rand(K, generator=g)
        self.gb_centers = centers
        self.gb_amps    = amps
        self.gb_sigmas  = sigmas

        # am_sin 파라미터(단일 샘플)
        self.am_carrier_w = 2.3
        self.am_mod_w     = 0.55
        self.am_mod_w2    = 0.18
        self.am_mod_depth  = 0.95
        self.am_mod_depth2 = 0.65
        self.am_phase = 2.0 * math.pi * torch.rand((), generator=g).item()

        # ─────────────────────────────────────────────────────────
        # (1) x 생성
        # ─────────────────────────────────────────────────────────
        if self.grid_type == 'uniform':
            x_1d = torch.linspace(self._xmin, self._xmax, self.num_points)
            self.x = x_1d.unsqueeze(0).repeat(self.num_data, 1)  # (B,N)
        else:
            # random grid: 각 샘플마다 다른 x를 저장하지 않고 __getitem__에서 생성
            # placeholder
            self.x = None

        # ─────────────────────────────────────────────────────────
        # (2) y 생성 (uniform grid에서만 사전 생성)
        # ─────────────────────────────────────────────────────────
        if self.grid_type == 'uniform':
            if self.func_type == 'doppler':
                # doppler는 "함수별 상수 오프셋" b만 추가
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
                    a_choices = self.linear_coeffs
                    a_idx = torch.randint(low=0, high=a_choices.numel(), size=(self.num_data, 1))
                    a = a_choices[a_idx].repeat(1, self.num_points)
                    y = a * (self.x) + eps

                elif self.func_type == 'sin':
                    y = torch.sin(self.x) + eps

                elif self.func_type == 'sinc':
                    y = torch.sinc(self.x) + eps

                elif self.func_type == 'gaussian_bumps':
                    # 단일 샘플 base function을 (1,N)에서 만들고 (B,N)로 반복
                    x0 = self.x[0:1, :]  # (1,N)
                    centers = self.gb_centers.view(1, 1, -1)  # (1,1,K)
                    amps    = self.gb_amps.view(1, 1, -1)
                    sigmas  = self.gb_sigmas.view(1, 1, -1)

                    diff = (x0.unsqueeze(-1) - centers) / sigmas          # (1,N,K)
                    y0   = (amps * torch.exp(-0.5 * diff.pow(2))).sum(-1) # (1,N)
                    y = y0.repeat(self.num_data, 1) + eps

                elif self.func_type == 'am_sin':
                    x0 = self.x
                    envelope = (
                        1.0 + self.am_mod_depth * torch.cos(self.am_mod_w * x0 + 0.3)
                    ) * (
                        1.0 + self.am_mod_depth2 * torch.cos(self.am_mod_w2 * x0 - 1.1)
                    )
                    y0 = envelope * torch.sin(self.am_carrier_w * x0 + self.am_phase)
                    taper = torch.exp(-0.5 * (x0 / 8.0).pow(2)) * 0.45 + 0.55
                    y = y0 * taper + eps

                else:
                    # circle
                    a = (torch.randint(low=0, high=2, size=(self.num_data, 1)) * 2 - 1).repeat(1, self.num_points)
                    r_choices = self.circle_radii
                    r_idx = torch.randint(low=0, high=r_choices.numel(), size=(self.num_data, 1))
                    r = r_choices[r_idx].repeat(1, self.num_points)
                    y = a * torch.sqrt((r ** 2 - self.x ** 2).clamp_min(0.0)) + eps

            self.dataset = y  # (B,N)

            # Z-normalize (전 데이터 mean/std)
            self.mean = self.dataset.mean()
            self.std  = self.dataset.std().clamp_min(1e-8)
            self.dataset = (self.dataset - self.mean) / self.std

        else:
            # random grid에서는 dataset을 미리 만들지 않음
            self.dataset = None
            # mean/std는 대략 0/1로 두되, inverse_transform은 호출되지 않는다고 가정
            self.mean = torch.tensor(0.0)
            self.std  = torch.tensor(1.0)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int):
        if self.grid_type == 'random':
            # 샘플마다 랜덤 x 생성
            if self.func_type == 'doppler':
                x_item = torch.rand(self.num_points)  # [0,1]
            else:
                x_item = (torch.rand(self.num_points) * 20.0) - 10.0  # [-10,10]
            x_item, _ = torch.sort(x_item)

            # y 생성
            if self.func_type == 'doppler':
                # doppler는 "함수별 상수 오프셋" b만 추가
                b = torch.randn(1).repeat(self.num_points) * self.noise_std

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
                    a_choices = self.linear_coeffs
                    a_idx = torch.randint(low=0, high=a_choices.numel(), size=(1,))
                    a = a_choices[a_idx].item()
                    y_item = a * (x_item) + eps

                elif self.func_type == 'sin':
                    y_item = torch.sin(x_item) + eps

                elif self.func_type == 'sinc':
                    y_item = torch.sinc(x_item) + eps

                elif self.func_type == 'gaussian_bumps':
                    dev = x_item.device
                    centers = self.gb_centers.to(dev).view(1, -1)  # (1,K)
                    amps    = self.gb_amps.to(dev).view(1, -1)
                    sigmas  = self.gb_sigmas.to(dev).view(1, -1)
                    diff = (x_item.view(-1, 1) - centers) / sigmas  # (N,K)
                    y0 = (amps * torch.exp(-0.5 * diff.pow(2))).sum(-1)  # (N,)
                    y_item = y0 + eps

                elif self.func_type == 'am_sin':
                    envelope = (
                        1.0 + self.am_mod_depth * torch.cos(self.am_mod_w * x_item + 0.3)
                    ) * (
                        1.0 + self.am_mod_depth2 * torch.cos(self.am_mod_w2 * x_item - 1.1)
                    )
                    y0 = envelope * torch.sin(self.am_carrier_w * x_item + self.am_phase)
                    taper = torch.exp(-0.5 * (x_item / 8.0).pow(2)) * 0.45 + 0.55
                    y_item = y0 * taper + eps

                else:
                    # circle
                    a = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
                    r_choices = self.circle_radii
                    r = r_choices[torch.randint(low=0, high=r_choices.numel(), size=(1,))].item()
                    y_item = a * torch.sqrt((r ** 2 - x_item ** 2).clamp_min(0.0)) + eps

            # random grid는 정규화하지 않고 raw로 반환(러너에서 처리)
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
            a_choices = self.linear_coeffs.to(dev)
            a_idx = torch.randint(low=0, high=a_choices.numel(), size=(B, 1), device=dev)
            a = a_choices[a_idx].repeat(1, N)
            y_raw = a * (x_batch) + eps
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













