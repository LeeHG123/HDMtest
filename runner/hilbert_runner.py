# runner/hilbert_runner.py
import os
import logging
from scipy.spatial import distance
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import math

from evaluate.power import calculate_ci
from evaluate.moments import moments_metrics
from datasets import data_scaler, data_inverse_scaler

from collections import OrderedDict

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from models import *

from functions.utils import *
from functions.loss import hilbert_loss_fn
from functions.sde import VPSDE1D
from functions.sampler import sampler
from functions.tsit5_sampler import sample_probability_flow_ode as tsit5_sample_ode

torch.autograd.set_detect_anomaly(True)


def _inv_softplus_scalar(y: float, eps: float = 1e-12) -> float:
    y = max(float(y), eps)
    return float(math.log(math.expm1(y)))


def _inv_softplus(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    t = t.clamp_min(eps)
    return torch.log(torch.expm1(t))


def kernel_se(x1, x2, hyp={"gain": 1.0, "len": 1.0}):
    """Squared-exponential kernel function"""
    x1_scaled = x1 / hyp["len"]
    x2_scaled = x2 / hyp["len"]
    D = torch.cdist(x1_scaled, x2_scaled, p=2.0).pow(2)  # sqeuclidean
    K = hyp["gain"] * torch.exp(-D)
    return K.to(torch.float64)

class HilbertNoise:
    """
    SE 커널 기반 힐버트 노이즈.
    - K(XX) 대칭화+지터 후 float64 eigh → M=E Λ^{1/2}, EΛ^{-1/2} 캐시
    - free_sample(Y) = K(Y,X) (EΛ^{-1/2}) z
    """

    def __init__(
        self,
        x_coords: torch.Tensor,
        *,
        hyp_len: float = 1.0,
        hyp_gain: float = 1.0,
        num_basis: int | None = None,
        jitter: float = 1e-6,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.jitter = float(jitter)
        self.device = device if device is not None else (
            x_coords.device if torch.is_tensor(x_coords) else torch.device("cpu")
        )

        # 좌표/정규화 캐시
        self.x = torch.as_tensor(x_coords, device=self.device, dtype=torch.float64).view(-1)  # (N,)
        self.N = int(self.x.numel())

        # SE 하이퍼
        self.hyp = {"gain": float(hyp_gain), "len": float(hyp_len)}

        # K(XX) 구성 → M, EΛ^{-1/2} 캐시
        self._build_eigendecomp(num_basis=num_basis)

    # ─────────────────────────────────────────────────────────
    # K(XX) → 고유분해 캐시
    # ─────────────────────────────────────────────────────────
    def _build_K_xx(self) -> torch.Tensor:
        X = self.x.view(-1, 1).to(torch.float64)
        K = kernel_se(X, X, self.hyp)

        # 대칭화 + 지터
        K = 0.5 * (K + K.T)
        K = K + (self.jitter * torch.eye(K.shape[0], dtype=K.dtype, device=K.device))
        return K.to(torch.float64)

    def _build_eigendecomp(self, num_basis: int | None = None):
        K = self._build_K_xx()  # float64
        eig_val, eig_vec = torch.linalg.eigh(K)  # 오름차순
        if num_basis is not None and 0 < num_basis < eig_val.numel():
            self.full_eig_val = eig_val
            self.full_eig_vec = eig_vec
            eig_val = eig_val[-num_basis:]
            eig_vec = eig_vec[:, -num_basis:]

        self.num_basis = int(eig_val.numel())
        self.eig_val = eig_val
        self.eig_vec = eig_vec

        # ★ 캐시를 float64로 유지 (수치 안정성↑), 필요 시 반환단에서만 float32로 변환
        Λ_sqrt = torch.sqrt(eig_val.clamp_min(0.0))
        Λ_isqrt = 1.0 / torch.sqrt(eig_val.clamp_min(1e-8))
        self.M = (eig_vec @ torch.diag(Λ_sqrt)).to(torch.float64)
        self.E_inv_sqrt = (eig_vec @ torch.diag(Λ_isqrt)).to(torch.float64)

    # ─────────────────────────────────────────────────────────
    # 샘플링 API
    # ─────────────────────────────────────────────────────────
    def sample(self, size):
        """
        size: (B, N) 형태 기대 — grid_dim=N
        return: (B, N) float32
        """
        B = int(size[0])
        z = torch.randn(B, self.num_basis, device=self.M.device, dtype=self.M.dtype)  # float64
        out64 = z @ self.M.T  # float64
        return out64.to(torch.float32)  # (B, N) float32

    def _K_yx(self, y_coords: torch.Tensor) -> torch.Tensor:
        """
        교차 커널 K(Y,X)  — y_coords: (B,N_y) or (N_y,)
        """
        if y_coords.dim() == 2:
            assert y_coords.size(0) == 1, "batch별 서로 다른 좌표는 free_sample에서 루프 처리합니다."
            y = y_coords.view(-1)
        else:
            y = y_coords.view(-1)
        y = y.to(self.device, dtype=torch.float64)
        Y = y.view(-1, 1)
        X = self.x.view(-1, 1)
        K_yx = kernel_se(Y, X, self.hyp)
        return K_yx.to(torch.float64)

    def free_sample(self, free_input: torch.Tensor) -> torch.Tensor:
        """
        free_input: (B, N_free)  — 좌표는 원좌표계 (정규화 전)
        return: (B, N_free) float32
        """
        device = free_input.device
        B, Ny = free_input.shape
        out64 = torch.zeros(B, Ny, device=device, dtype=torch.float64)
        E_inv_sqrt = self.E_inv_sqrt.to(device, dtype=torch.float64)

        for i in range(B):
            y = free_input[i].to(torch.float64)
            K_yx = self._K_yx(y)  # (Ny, N) float64
            z = torch.randn(self.num_basis, 1, device=device, dtype=torch.float64)
            f_y = K_yx @ (E_inv_sqrt @ z)  # (Ny,1) float64
            out64[i] = f_y.view(-1)
        return out64.to(torch.float32)

    @torch.no_grad()
    def sample_latent(self, B: int) -> torch.Tensor:
        """
        잠재 z ~ N(0, I) 를 (B, num_basis) 형태의 float64로 반환.
        M = E Λ^{1/2},  E_inv_sqrt = E Λ^{-1/2} 캐시와 호환되도록 dtype을 float64로 둔다.
        """
        return torch.randn(B, self.num_basis, device=self.M.device, dtype=torch.float64)

    @torch.no_grad()
    def project(self, y_coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        e(Y) = K(Y, X) E Λ^{-1/2} z  를 계산하여 좌표 Y로의 힐버트 노이즈를 생성한다.
        y_coords : (B, N_y) 또는 (N_y,)
        z        : (B, num_basis) 또는 (num_basis,)
        return   : (B, N_y) float32
        """
        # 배치화 정리
        if y_coords.dim() == 1:
            y_coords = y_coords.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(y_coords.size(0), -1)

        B, Ny = y_coords.shape
        out64 = torch.zeros(B, Ny, device=y_coords.device, dtype=torch.float64)
        E_inv_sqrt = self.E_inv_sqrt.to(self.device, dtype=torch.float64)  # (N, m)

        for b in range(B):
            y = y_coords[b].to(torch.float64)
            K_yx = self._K_yx(y)  # (Ny, N) float64
            coef = E_inv_sqrt @ z[b].view(-1, 1)  # (N,1)
            f_y = K_yx @ coef  # (Ny,1)
            out64[b] = f_y.view(-1)

        return out64.to(torch.float32)


class HilbertDiffusion(object):
    def __init__(self, args, config, dataset, test_dataset, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        x_attr = getattr(dataset, "x", None)

        if torch.is_tensor(x_attr):
            # dataset.x can be (B,N) or (N,)
            grid_coords = x_attr[0] if x_attr.dim() == 2 else x_attr
        else:
            # Prefer a deterministic uniform grid if domain bounds are available
            xmin = getattr(dataset, "_xmin", None)
            xmax = getattr(dataset, "_xmax", None)

            if xmin is not None and xmax is not None:
                N = int(getattr(dataset, "num_points", getattr(config.data, "dimension", 256)))
                grid_coords = torch.linspace(float(xmin), float(xmax), N)
            else:
                # Fallback: take coordinates from one sample
                x0, _ = dataset[0]
                grid_coords = x0

        # Ensure shape (N,) on device, sorted
        if grid_coords.dim() >= 2:
            grid_coords = grid_coords.squeeze(-1).squeeze(0)
        grid_coords = grid_coords.to(self.device).view(-1).sort().values
        num_basis = getattr(config.data, "num_basis", None)

        # 좌표 정규화 파라미터(데이터셋에서 읽어 러너/커널 일관 유지)
        coord_scale = getattr(dataset, "coord_scale", 1.0)
        coord_offset = getattr(dataset, "coord_offset", 0.0)

        # (기본값) SE 커널 — 기존과 동일
        kernel_type = "se"
        self.W = HilbertNoise(
            x_coords=grid_coords,
            hyp_len=config.data.hyp_len,
            hyp_gain=config.data.hyp_gain,
            num_basis=num_basis,
            device=self.device,
        )
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        self.sde = VPSDE1D(schedule="cosine")
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.spec_lambda = float(getattr(config.training, "spec_lambda", 0.05))

        # --- 야코비안 보정 상수 주입 ---
        # 좌표 정규화: u = (x - offset) / scale  =>  dx = scale * du
        # measure_scale := scale = dataset.coord_scale
        try:
            ms = float(getattr(dataset, "coord_scale", 1.0))
        except Exception:
            ms = 1.0
        setattr(self.config.model, "measure_scale", ms)

    def _coord_norm(self, x: torch.Tensor) -> torch.Tensor:
        coord_scale = getattr(self.dataset, "coord_scale", 10.0)
        coord_offset = getattr(self.dataset, "coord_offset", 0.0)
        return (x - coord_offset) / coord_scale

    def validate(self, model, val_loader, tb_logger, step, calc_fixed_grid_loss: bool = True):
        """
        Validation 함수. Resolution-free 손실과 고정 그리드 손실을 계산합니다.
        """
        model.eval()

        res_free_points = self.args.res_free_points
        val_losses_res_free = {n_res: [] for n_res in res_free_points}

        if calc_fixed_grid_loss:
            val_losses_fixed = []

        with torch.no_grad():
            for i, (x_fixed, y_fixed) in enumerate(val_loader):
                if i >= 10:
                    break

                B = y_fixed.shape[0]

                # --- 1. Resolution-Free 검증 (여러 해상도에 대해 반복) ---
                if not self.args.disable_resolution_free:
                    for N_res in res_free_points:
                        # 데이터셋 도메인에서 무작위 정렬 좌표 샘플
                        xmin = float(getattr(self.dataset, "_xmin", -10.0))
                        xmax = float(getattr(self.dataset, "_xmax", 10.0))

                        x_res_free = (
                            torch.rand(B, N_res, device=self.device) * (xmax - xmin) + xmin
                        ).sort(dim=1).values

                        y_raw = self.dataset.generate_raw(x_res_free, device=self.device)
                        y_res_free = (y_raw - self.dataset.mean.to(self.device)) / self.dataset.std.to(
                            self.device
                        )

                        # 손실 계산 (좌표 정규화 포함)
                        x_coord_norm_res_free = self._coord_norm(x_res_free)
                        t = torch.rand(B, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                        e = self.W.free_sample(x_res_free).to(self.device)

                        loss_res_free = hilbert_loss_fn(
                            model,
                            self.sde,
                            y_res_free,
                            t,
                            e,
                            x_coord_norm_res_free,
                            global_step=step,
                            max_steps=getattr(self, "_max_steps", None),
                        )
                        val_losses_res_free[N_res].append(loss_res_free.item())

                # --- 2. 고정 그리드 검증 (훈련 통계로 재정규화) ---
                if calc_fixed_grid_loss:
                    x_fixed_dev = x_fixed.to(self.device).squeeze(-1)  # (B, N)
                    y_fixed_dev = y_fixed.to(self.device).squeeze(-1)  # (B, N) -- test 통계로 정규화된 상태
                    y_fixed_raw = self.test_dataset.inverse_transform(y_fixed_dev)
                    y_fixed_train_norm = (y_fixed_raw - self.dataset.mean.to(self.device)) / self.dataset.std.to(
                        self.device
                    )
                    x_coord_norm_fixed = self._coord_norm(x_fixed_dev)
                    t = torch.rand(B, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                    e = self.W.free_sample(x_fixed_dev).to(self.device)

                    loss_fixed = hilbert_loss_fn(
                        model,
                        self.sde,
                        y_fixed_train_norm,
                        t,
                        e,
                        x_coord_norm_fixed,
                        global_step=step,
                        max_steps=getattr(self, "_max_steps", None),
                    )
                    val_losses_fixed.append(loss_fixed.item())

        # --- 결과 계산 및 로깅 ---
        for N_res, losses in val_losses_res_free.items():
            if losses:
                avg_val_loss = np.mean(losses)
                tb_logger.add_scalar(f"val_loss/resolution_free_{N_res}", avg_val_loss, global_step=step)

        avg_val_loss_fixed = None
        if calc_fixed_grid_loss and val_losses_fixed:
            avg_val_loss_fixed = np.mean(val_losses_fixed)
            tb_logger.add_scalar("val_loss/fixed_grid", avg_val_loss_fixed, global_step=step)

        model.train()

        if val_losses_res_free and res_free_points:
            first_n_res = res_free_points[0]
            if val_losses_res_free[first_n_res]:
                return np.mean(val_losses_res_free[first_n_res])
        return avg_val_loss_fixed

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if args.distributed:
            sampler_ = DistributedSampler(
                self.dataset, shuffle=True, seed=args.seed if args.seed is not None else 0
            )
        else:
            sampler_ = None
        train_loader = data.DataLoader(
            self.dataset, batch_size=config.training.batch_size, num_workers=config.data.num_workers, sampler=sampler_
        )
        steps_per_epoch = len(train_loader)
        self._max_steps = steps_per_epoch * self.config.training.n_epochs

        # Validation loader
        val_loader = data.DataLoader(
            self.test_dataset,
            batch_size=config.training.val_batch_size,
            num_workers=config.data.num_workers,
            shuffle=False,
        )

        # Model
        if config.model.model_type == "ddpm_mnist":
            model = Unet(
                dim=config.data.image_size,
                channels=config.model.channels,
                dim_mults=config.model.dim_mults,
                is_conditional=config.model.is_conditional,
            )
        elif config.model.model_type == "FNO":
            model = FNO(
                n_modes=config.model.n_modes,
                hidden_channels=config.model.hidden_channels,
                in_channels=config.model.in_channels,
                out_channels=config.model.out_channels,
                lifting_channels=config.model.lifting_channels,
                projection_channels=config.model.projection_channels,
                n_layers=config.model.n_layers,
                joint_factorization=config.model.joint_factorization,
                norm=config.model.norm,
                preactivation=config.model.preactivation,
                separable=config.model.separable,
            )
        elif config.model.model_type == "KNO":
            from models import KNO
            model = KNO(config)
        elif config.model.model_type == "MHLKNO":
            from models import MHLKNO
            model = MHLKNO(config)
        elif config.model.model_type == "ChebyshevMHLKNO":
            from models import ChebyshevMHLKNO
            model = ChebyshevMHLKNO(config)
        elif config.model.model_type == "MHLKNO_LINATTN":
            from models import MHLKNO_LinAttn
            model = MHLKNO_LinAttn(config)

        model = model.to(self.device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

        logging.info("Model loaded.")

        # ---------- Optimizer: 파라미터 그룹핑 최적화 ----------
        base_lr = config.optim.lr
        # 주파수 관련 파라미터에 적용할 배수 (config의 kappa_lr_multiplier를 그대로 재사용)
        kappa_lr_mul = float(getattr(config.optim, "kappa_lr_multiplier", 1.0))

        mm_ref = model.module if hasattr(model, "module") else model

        # 파라미터 그룹 리스트 초기화
        # (MHLKNO(+LINATTN)) 주파수/커널 관련 그룹
        mhl_omega_params = []  # MHLKNO RFF/커널 주파수 관련 파라미터

        # KNO/GSM 전용 그룹
        kno_amp_params = []    # log_gain (RBF), log_w (GSM)
        kno_bw_params = []     # log_len (RBF), log_sig (GSM)
        kno_freq_params = []   # log_mu (GSM)
        kno_feat_params = []   # NS-GSM Networks
        kno_tcond_params = []  # Time conditioning

        other_params = []      # Weights, Biases, Projections

        for n, p in mm_ref.named_parameters():
            if not p.requires_grad:
                continue

            # 1) MHLKNO / MHLKNO_LINATTN: 커널 주파수 계열(길이/대역폭/중심주파수)
            if getattr(config.model, "model_type", "") in ("MHLKNO", "MHLKNO_LINATTN"):
                if ("log_len" in n) or ("log_sig" in n) or ("log_mu" in n):
                    mhl_omega_params.append(p)
                    continue

            # 2) KNO / MHLKNO 커널 파라미터
            #    - KNO:  spectral_blocks.*.kern.*
            #    - MHLKNO: layers.*.kernel.*
            is_kernel_param = (".kern." in n) or (".kernel." in n)
            if is_kernel_param:
                # amplitude 계열: log_gain (RBF), log_w (GSM)
                if n.endswith(".log_gain") or n.endswith(".log_w"):
                    kno_amp_params.append(p)
                    continue
                # bandwidth 계열: log_len (RBF), log_sig (GSM)
                if n.endswith(".log_len") or n.endswith(".log_sig"):
                    kno_bw_params.append(p)
                    continue
                # 중심 주파수 계열: log_mu (GSM)
                if n.endswith(".log_mu"):
                    kno_freq_params.append(p)
                    continue
                # NS-GSM feature network
                if (".kern.feat." in n) or (".kernel.feat." in n):
                    kno_feat_params.append(p)
                    continue
                # time-conditioning MLP (KNO / MHLKNO 공통)
                if (".kern.tmlp." in n) or (".kernel.tmlp." in n):
                    kno_tcond_params.append(p)
                    continue

            # 3) 기타 파라미터 (Conv, Linear, Norm 등)
            other_params.append(p)

        # Optimizer 그룹 구성
        param_groups = [
            {"params": other_params, "lr": base_lr, "weight_decay": 0.01},  # 일반 가중치 decay
        ]

        # MHLKNO Omega 그룹
        if mhl_omega_params:
            lr_mult = kappa_lr_mul if kappa_lr_mul > 0 else 1.0
            param_groups.append({"params": mhl_omega_params, "lr": base_lr * lr_mult, "weight_decay": 0.0})
            if args.local_rank == 0:
                logging.info(f"[MHLKNO] Omega-like params grouped with LR x{lr_mult} and WD=0.0")

        # KNO Groups
        if kno_amp_params:
            param_groups.append({"params": kno_amp_params, "lr": base_lr * 1.00, "weight_decay": 1e-4})
        if kno_bw_params:
            param_groups.append({"params": kno_bw_params, "lr": base_lr * 0.50, "weight_decay": 0.0})
        if kno_freq_params:
            param_groups.append({"params": kno_freq_params, "lr": base_lr * 0.25, "weight_decay": 0.0})
        if kno_feat_params:
            param_groups.append({"params": kno_feat_params, "lr": base_lr * 0.50, "weight_decay": 0.0})
        if kno_tcond_params:
            param_groups.append({"params": kno_tcond_params, "lr": base_lr * 0.50, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(param_groups, amsgrad=True)

        start_epoch, step = 0, 0
        for epoch in range(config.training.n_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device).squeeze(-1)   # (B, N)
                y = y.to(self.device).squeeze(-1)   # (B, N)
                x_coord_norm = self._coord_norm(x)  # (B, N)  in [-1,1]

                data_time += time.time() - data_start
                model.train()
                step += 1

                if config.data.dataset == "Melbourne":
                    y = data_scaler(y)

                t = torch.rand(y.shape[0], device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                # 좌표가 랜덤이면 해당 좌표에 맞춘 Hilbert noise
                if getattr(self.config.data, "grid_type", "uniform") in ("random", "adapted_random"):
                    e = self.W.free_sample(x).to(self.device)
                else:
                    e = self.W.sample(y.shape).to(self.device).squeeze(-1)

                # 스코어 학습 손실 (좌표 가중 포함)
                loss_score = hilbert_loss_fn(
                    model,
                    self.sde,
                    y,
                    t,
                    e,
                    x_coord_norm,
                    global_step=step,
                    max_steps=getattr(self, "_max_steps", None),
                ).to(self.device)

                loss = loss_score

                # 로깅(분해)
                tb_logger.add_scalar("loss/data", float(loss_score.detach()), step)
                tb_logger.add_scalar("train_loss", float(torch.abs(loss).detach()), step)

                optimizer.zero_grad()
                loss.backward()

                # ----- Monitoring (minimal) -----
                if step % 100 == 0:
                    with torch.no_grad():
                        mm = model.module if hasattr(model, "module") else model

                        # (기존 유지) MHLKNO 계열 omega 모니터링(있을 때만)
                        if config.model.model_type == "MHLKNO" and hasattr(mm, "layers"):
                            for li, layer in enumerate(mm.layers):
                                if hasattr(layer, "omega"):
                                    omega_now = layer.omega.detach()
                                    tb_logger.add_scalar(f"omega/layer{li}_mean", float(omega_now.mean()), step)
                                    tb_logger.add_scalar(f"omega/layer{li}_std", float(omega_now.std()), step)

                if args.local_rank == 0:
                    logging.info(
                        f"step: {step}, loss: {float(torch.abs(loss).detach())}, data time: {data_time / (i+1)}"
                    )

                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                except Exception:
                    pass

                optimizer.step()

                # Validation
                if step % config.training.val_freq == 0 and step > 0:
                    val_loss = self.validate(model, val_loader, tb_logger, step, calc_fixed_grid_loss=True)
                    if args.local_rank == 0:
                        logging.info(f"step: {step}, val_loss (res-free): {val_loss}")
                        if step % (config.training.val_freq * 10) == 0:
                            logging.info(f"Generating validation loss plots at step {step}...")
                            self._plot_validation_losses(tb_log_dir=tb_logger.log_dir)

                if step % config.training.ckpt_store == 0:
                    self.ckpt_dir = os.path.join(args.log_path, f"ckpt_step_{step}.pth")
                    torch.save(model.state_dict(), self.ckpt_dir)
                    latest_ckpt_dir = os.path.join(args.log_path, "ckpt.pth")
                    torch.save(model.state_dict(), latest_ckpt_dir)

                data_start = time.time()

    def _plot_validation_losses(self, tb_log_dir: str | None = None):
        """
        TensorBoard 로그에서 검증 손실을 읽어와 **개별 그래프**로 시각화하고 저장합니다.
        - 'val_loss/fixed_grid'
        - 'val_loss/resolution_free_...'
        """
        if hasattr(self.config, "tb_logger") and hasattr(self.config.tb_logger, "log_dir"):
            tb_log_dir = self.config.tb_logger.log_dir
        else:
            tb_log_dir = os.path.join(self.args.exp, "tensorboard", self.args.doc)

        if not os.path.exists(tb_log_dir):
            logging.warning(f"TensorBoard log directory not found: {tb_log_dir}")
            return

        try:
            event_files = [
                os.path.join(tb_log_dir, f)
                for f in os.listdir(tb_log_dir)
                if "events.out.tfevents" in f
            ]
            if not event_files:
                raise IndexError
            event_file = sorted(event_files, key=os.path.getmtime)[-1]
        except IndexError:
            logging.warning(f"No TensorBoard event file found in {tb_log_dir}")
            return

        logging.info(f"Reading TensorBoard logs from: {event_file}")
        ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]

        plot_save_dir = self.args.log_path

        for tag in sorted(tags):
            if tag == "val_loss/fixed_grid" or tag.startswith("val_loss/resolution_free_"):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                if not steps:
                    continue

                plt.figure(figsize=(10, 6))

                if tag == "val_loss/fixed_grid":
                    plot_label = "Fixed-Grid Val Loss"
                    plot_title = "Fixed-Grid Validation Loss over Training"
                    plot_color = "crimson"
                else:
                    try:
                        points = tag.split("_")[-1]
                        plot_label = f"Res-free ({points} points)"
                        plot_title = f"Resolution-Free Validation Loss ({points} points)"
                        plot_color = "royalblue"
                    except (IndexError, ValueError):
                        plot_label = tag
                        plot_title = f"Validation Loss for {tag}"
                        plot_color = "darkslateblue"

                plt.plot(steps, values, label=plot_label, color=plot_color)
                plt.xlabel("Training Steps")
                plt.ylabel("Loss")
                plt.title(plot_title)
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)

                filename = tag.replace("/", "_") + ".png"
                save_path = os.path.join(plot_save_dir, f"validation_loss_{filename}")

                plt.savefig(save_path)
                plt.close()
                logging.info(f"Saved validation loss plot to {save_path}")

    def sample(self, score_model=None):
        args, config = self.args, self.config
        self._plot_validation_losses()

        if config.model.model_type == "ddpm_mnist":
            model = Unet(
                dim=config.data.image_size,
                channels=config.model.channels,
                dim_mults=config.model.dim_mults,
                is_conditional=config.model.is_conditional,
            )
        elif config.model.model_type == "FNO":
            model = FNO(
                n_modes=config.model.n_modes,
                hidden_channels=config.model.hidden_channels,
                in_channels=config.model.in_channels,
                out_channels=config.model.out_channels,
                lifting_channels=config.model.lifting_channels,
                projection_channels=config.model.projection_channels,
                n_layers=config.model.n_layers,
                joint_factorization=config.model.joint_factorization,
                norm=config.model.norm,
                preactivation=config.model.preactivation,
                separable=config.model.separable,
            )
        elif config.model.model_type == "KNO":
            from models import KNO
            model = KNO(config)
        elif config.model.model_type == "MHLKNO":
            from models import MHLKNO
            model = MHLKNO(config)
        elif config.model.model_type == "ChebyshevMHLKNO":
            from models import ChebyshevMHLKNO
            model = ChebyshevMHLKNO(config)
        elif config.model.model_type == "MHLKNO_LINATTN":
            from models import MHLKNO_LinAttn
            model = MHLKNO_LinAttn(config)

        model = model.to(self.device)

        if score_model is not None:
            model = score_model

        elif ("ckpt_dir" in config.model.__dict__.keys()):
            # Check if specific checkpoint step is requested
            if args.ckpt_step is not None:
                ckpt_path = os.path.join(args.log_path, f"ckpt_step_{args.ckpt_step}.pth")
                if os.path.exists(ckpt_path):
                    ckpt_dir = ckpt_path
                    logging.info(f"Using checkpoint from step {args.ckpt_step}: {ckpt_path}")
                else:
                    logging.warning(f"Checkpoint for step {args.ckpt_step} not found: {ckpt_path}")
                    logging.info("Falling back to latest checkpoint")
                    ckpt_path = os.path.join(args.log_path, "ckpt.pth")
                    if os.path.exists(ckpt_path):
                        ckpt_dir = ckpt_path
                    else:
                        ckpt_dir = config.model.ckpt_dir
            else:
                # First try the latest checkpoint from training
                ckpt_path = os.path.join(args.log_path, "ckpt.pth")
                if os.path.exists(ckpt_path):
                    ckpt_dir = ckpt_path
                else:
                    ckpt_dir = config.model.ckpt_dir

            if os.path.exists(ckpt_dir):
                states = torch.load(ckpt_dir, map_location=config.device)
                if args.distributed:
                    state_dict = OrderedDict()
                    for k, v in states.items():
                        name = k[7:] if k.startswith("module.") else k
                        state_dict[name] = v
                    model.load_state_dict(state_dict, strict=False)
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
                else:
                    model.load_state_dict(states, strict=False)
            else:
                logging.warning(
                    f"Checkpoint not found: {ckpt_dir} — skipping load (model_type={self.config.model.model_type})."
                )

        logging.info("Done loading model")
        model.eval()

        enable_clip = getattr(config.sampling, "enable_rms_clip", False)
        clip_threshold = getattr(config.sampling, "rms_clip_threshold", None)
        if enable_clip:
            logging.info(f"RMS clipping enabled with threshold: {clip_threshold}")

        test_loader = torch.utils.data.DataLoader(self.test_dataset, config.sampling.batch_size, shuffle=False)

        x_0, y_0 = next(iter(test_loader))
        y_0 = y_0.squeeze(-1)  # (B, N)

        if self.args.disable_resolution_free:
            free_input = x_0.squeeze(-1)  # 고정 그리드
            if str(config.data.dataset).lower() in (
                "quadratic",
                "linear",
                "circle",
                "sin",
                "sinc",
                "doppler",
                "gaussianbumps",
                "amsin",
            ):
                y00 = self.test_dataset.inverse_transform(y_0)
            else:
                y00 = y_0  # fallback
        else:
            N_res = self.args.res_free_points[0]
            xmin = float(getattr(self.dataset, "_xmin", -10.0))
            xmax = float(getattr(self.dataset, "_xmax", 10.0))

            # 1) 균등 공통 그리드(권장)
            grid_common = torch.linspace(xmin, xmax, N_res, device=self.device)

            # 배치 전체에 공유
            free_input = grid_common.unsqueeze(0).expand(config.sampling.batch_size, -1).contiguous()

            # 원스케일 GT 생성(배치×공통그리드)
            y00 = self.dataset.generate_raw(free_input, device=self.device)

        y_shape = (config.sampling.batch_size, config.data.dimension)

        if self.args.sample_type in ["srk", "sde"]:  # SRK/Euler 케이스
            with torch.no_grad():
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                if self.args.disable_resolution_free:
                    y = self.W.sample(y_shape).to(self.device) * self.sde.marginal_std(t)[:, None]
                else:
                    y = self.W.free_sample(free_input).to(self.device) * self.sde.marginal_std(t)[:, None]
                free_input_norm = self._coord_norm(free_input.to(self.device))
                y = sampler(
                    y,
                    free_input_norm,
                    model=model,
                    sde=self.sde,
                    device=self.device,
                    W=self.W,
                    eps=self.sde.eps,
                    dataset=config.data.dataset,
                    steps=self.args.nfe,
                    method="srk",
                )

        elif self.args.sample_type == "tsit5_ode":
            with torch.no_grad():
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                if self.args.disable_resolution_free:
                    yT = self.W.sample(y_shape).to(self.device)
                else:
                    yT = self.W.free_sample(free_input).to(self.device)
                yT = yT / (torch.std(yT, dim=1, keepdim=True) + 1e-12)
                yT = yT * self.sde.marginal_std(t)[:, None]
                free_input_norm = self._coord_norm(free_input.to(self.device))
                # ① 샘플 생성용(시각화/저장): RMS clip 적용
                y_gen = tsit5_sample_ode(
                    model,
                    self.sde,
                    x_t0=yT,
                    x_coord=free_input_norm,
                    device=self.device,
                    inference_steps=self.args.nfe,
                    rtol=1e-5,
                    atol=1e-5,
                    enable_rms_clip=enable_clip,
                    rms_clip_threshold=clip_threshold,
                )
                # ② Power 계산용: RMS clip 비활성화
                y_pow = tsit5_sample_ode(
                    model,
                    self.sde,
                    x_t0=yT,
                    x_coord=free_input_norm,
                    device=self.device,
                    inference_steps=self.args.nfe,
                    rtol=1e-5,
                    atol=1e-5,
                    enable_rms_clip=False,
                    rms_clip_threshold=None,
                )

        # ──── Tsit5 결과 시각화 ────
        if self.args.sample_type == "tsit5_ode" and config.data.dataset in [
            "Quadratic",
            "Linear",
            "Circle",
            "Sin",
            "Sinc",
            "Doppler",
            "GaussianBumps",
            "AMSin",
        ]:
            x_0 = x_0.cpu()
            y0_plot = self.test_dataset.inverse_transform(y_0).cpu()
            # 시각화용(clip 적용 샘플)
            y_gen_plot = (y_gen * self.dataset.std.to(y_gen.device) + self.dataset.mean.to(y_gen.device)).cpu()
            # Power 계산용(무클립 샘플)
            y_pow_plot = (y_pow * self.dataset.std.to(y_pow.device) + self.dataset.mean.to(y_pow.device)).cpu()

            y_gt = y00.cpu()
            n_tests = y_pow_plot.shape[0] // 10
            power_res = calculate_ci(y_pow_plot, y_gt, n_tests=n_tests)
            print(f"[Tsit5] resolution-free power(avg 30) = {power_res}")
            m_err, c_err = moments_metrics(free_input.cpu(), y_pow_plot, y_gt)
            print(f"[Tsit5] moments (raw): mean_L2 = {m_err:.6e}, cov_HS = {c_err:.6e}")

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ds_name = str(config.data.dataset).lower()

            for i in range(min(10, y0_plot.shape[0])):
                ax[0].plot(x_0[i], y0_plot[i], color="k", alpha=0.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_gen_plot.shape[0]):
                ax[1].plot(free_input[i].cpu(), y_gen_plot[i], alpha=0.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"Tsit5-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_tsit5.png")
            print("Saved plot fig to visualization_tsit5.png")
            plt.clf()
            plt.figure()

        if self.args.sample_type == "srk":
            with torch.no_grad():
                y_shape = (config.sampling.batch_size, config.data.dimension)
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

            y0_plot = self.test_dataset.inverse_transform(y_0)
            y_plot = y * self.dataset.std.to(y.device) + self.dataset.mean.to(y.device)

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            ds_name = str(config.data.dataset).lower()

            for i in range(min(config.sampling.batch_size, y0_plot.shape[0])):
                ax[0].plot(x_0[i, :].cpu(), y0_plot[i, :].cpu())
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            # 우측: 생성 결과(기존 유지)
            for i in range(y.shape[0]):
                ax[1].plot(free_input[i, :].cpu(), y_plot[i, :].cpu(), alpha=1)

            # (원본 동작 유지) n_tests는 여기서 정의되어야 안정적이지만,
            # 기존 코드 흐름을 최대한 유지하기 위해 아래 라인은 원본대로 둡니다.
            # 필요시: n_tests = y_plot.shape[0] // 10
            print("Calculate Confidence Interval:")
            power_res = calculate_ci(y_plot, y0_plot, n_tests=n_tests)
            print(f"Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}")
            logging.info(f"Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}")
            ax[1].set_title(f"resolution-free, power(avg of 30 trials): {power_res}")

        else:
            y_0 = y_0.squeeze(-1)
            with torch.no_grad():
                for _ in tqdm(range(1), desc="Generating image samples"):
                    y_shape = (config.sampling.batch_size, config.data.dimension)
                    t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                    y = self.W.sample(y_shape).to(self.device) * self.sde.marginal_std(t)[:, None]
                    y = sampler(y, model, self.sde, self.device, self.W, self.sde.eps, config.data.dataset)

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            if config.data.dataset == "Melbourne":
                lp = 10
                n_tests = y.shape[0] // 10
                y = data_inverse_scaler(y)
            if config.data.dataset == "Gridwatch":
                lp = y.shape[0]
                n_tests = y.shape[0] // 10
                plt.ylim([-2, 3])

            for i in range(lp):
                ax[0].plot(x_0[i, :].cpu(), y[i, :].cpu())
                ax[1].plot(x_0[i, :].cpu(), y_0[i, :].cpu(), c="black", alpha=1)

            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(lp):
                ax[1].plot(x_0[i, :].cpu(), y[i, :].cpu(), alpha=1)

            power = calculate_ci(y, y_0, n_tests=n_tests)
            print(f"Calculate Confidence Interval: grid, 0th: {power}")

            ax[1].set_title(f"grid, power(avg of 30 trials):{power}")

        # Visualization figure save
        plt.savefig("visualization_default.png")
        print("Saved plot fig to {}".format("visualization_default.png"))
        plt.clf()
        plt.figure()

