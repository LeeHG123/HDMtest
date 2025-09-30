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

def kernel_se(x1, x2, hyp={'gain':1.0,'len':1.0}):
    """ Squared-exponential kernel function """
    x1_scaled = x1 / hyp['len']
    x2_scaled = x2 / hyp['len']
    D = torch.cdist(x1_scaled, x2_scaled, p=2.0).pow(2) # sqeuclidean
    K = hyp['gain'] * torch.exp(-D)
    return K.to(torch.float64)

class HilbertNoise:
    """
    SE 커널 기반 힐버트 노이즈.
    - K(XX) 대칭화+지터 후 float64 eigh → M=E Λ^{1/2}, EΛ^{-1/2} 캐시
    - free_sample(Y) = K(Y,X) (EΛ^{-1/2}) z
    """
    def __init__(self,
                 x_coords: torch.Tensor,
                 *,
                 hyp_len: float = 1.0,
                 hyp_gain: float = 1.0,
                 num_basis: int | None = None,
                 jitter: float = 1e-6,
                 device: torch.device | None = None):
        super().__init__()

        self.jitter = float(jitter)
        self.device = device if device is not None else (x_coords.device if torch.is_tensor(x_coords) else torch.device("cpu"))

        # 좌표/정규화 캐시
        self.x = torch.as_tensor(x_coords, device=self.device, dtype=torch.float64).view(-1)  # (N,)
        self.N = int(self.x.numel())

        # SE 하이퍼
        self.hyp = {'gain': float(hyp_gain), 'len': float(hyp_len)}

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
        K = self._build_K_xx()                                         # float64
        # eigh
        eig_val, eig_vec = torch.linalg.eigh(K)                        # 오름차순
        if num_basis is not None and 0 < num_basis < eig_val.numel():
            self.full_eig_val = eig_val
            self.full_eig_vec = eig_vec
            eig_val = eig_val[-num_basis:]
            eig_vec = eig_vec[:, -num_basis:]

        self.num_basis = int(eig_val.numel())
        self.eig_val = eig_val
        self.eig_vec = eig_vec

        # ★ 캐시를 float64로 유지 (수치 안정성↑), 필요 시 반환단에서만 float32로 변환
        Λ_sqrt  = torch.sqrt(eig_val.clamp_min(0.0))
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
            K_yx = self._K_yx(y)                       # (Ny, N) float64
            z = torch.randn(self.num_basis, 1, device=device, dtype=torch.float64)
            f_y = K_yx @ (E_inv_sqrt @ z)              # (Ny,1) float64
            out64[i] = f_y.view(-1)
        return out64.to(torch.float32)

class HilbertDiffusion(object):
    def __init__(self, args, config, dataset, test_dataset, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        grid_coords = dataset.x[0].to(self.device)              # (N,)
        num_basis = getattr(config.data, 'num_basis', None)

        # 좌표 정규화 파라미터(데이터셋에서 읽어 러너/커널 일관 유지)
        coord_scale  = getattr(dataset, 'coord_scale', 1.0)
        coord_offset = getattr(dataset, 'coord_offset', 0.0)

        # (기본값) SE 커널 — 기존과 동일
        kernel_type = "se"
        # 생성: SE 커널 전용
        self.W = HilbertNoise(
            x_coords=grid_coords,
            hyp_len=config.data.hyp_len,
            hyp_gain=config.data.hyp_gain,
            num_basis=num_basis,
            device=self.device,
        )
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        self.sde = VPSDE1D(schedule='cosine')
        self.dataset = dataset
        self.test_dataset = test_dataset       
        self.spec_lambda = float(getattr(config.training, "spec_lambda", 0.05))    

    def _coord_norm(self, x: torch.Tensor) -> torch.Tensor:
        coord_scale  = getattr(self.dataset, 'coord_scale', 10.0)
        coord_offset = getattr(self.dataset, 'coord_offset', 0.0)
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
                        x_res_free = (torch.rand(B, N_res, device=self.device) * 20 - 10).sort(dim=1).values

                        if self.config.data.dataset == 'Quadratic':
                            a   = torch.randint(low=0, high=2, size=(B, 1), device=self.device) * 2 - 1
                            eps = torch.randn(B, 1, device=self.device).repeat(1, N_res)
                            y_res_free = a * (x_res_free ** 2) + eps
                            # 훈련셋 통계로 Z-정규화
                            y_res_free = (y_res_free - self.dataset.mean.to(self.device)) / self.dataset.std.to(self.device)

                        elif self.config.data.dataset == 'Gaussian':
                            log_phi = -0.5 * x_res_free**2 - 0.5 * math.log(2 * math.pi)
                            eps = torch.randn_like(log_phi) * 1e-2
                            log_phi_noisy = log_phi + eps
                            y_res_free = (log_phi_noisy - self.dataset.mean.to(self.device)) / self.dataset.std.to(self.device)
                        elif self.config.data.dataset == 'Doppler': 
                            x_res_free = torch.rand(B, N_res, device=self.device).sort(dim=1).values
                            noise_std = float(getattr(self.dataset, "noise_std", 0.0))
                            b = (torch.randn(B, 1, device=self.device) * noise_std).repeat(1, N_res)
                            amp   = torch.sqrt((x_res_free * (1.0 - x_res_free)).clamp_min(0.0))
                            phase = (2.0 * math.pi * 1.05) / (x_res_free + 0.05)
                            y_raw = amp * torch.sin(phase) + b
                            y_res_free = (y_raw - self.dataset.mean.to(self.device)) / self.dataset.std.to(self.device)   
                        else:
                            y_res_free = None

                        if y_res_free is not None:
                            x_coord_norm_res_free = self._coord_norm(x_res_free)
                            t = torch.rand(B, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                            e = self.W.free_sample(x_res_free).to(self.device)

                            loss_res_free = hilbert_loss_fn(
                                model, self.sde, y_res_free, t, e, x_coord_norm_res_free,
                                global_step=step, max_steps=getattr(self, "_max_steps", None),
                                spec_lambda=self.spec_lambda,
                            )
                            val_losses_res_free[N_res].append(loss_res_free.item())

                # --- 2. 고정 그리드 검증 (훈련 통계로 재정규화) ---
                if calc_fixed_grid_loss:
                    x_fixed_dev = x_fixed.to(self.device).squeeze(-1)  # (B, N)
                    y_fixed_dev = y_fixed.to(self.device).squeeze(-1)  # (B, N)  -- test 통계로 정규화된 상태                   
                    y_fixed_raw = self.test_dataset.inverse_transform(y_fixed_dev)  # 원스케일
                    y_fixed_train_norm = (y_fixed_raw - self.dataset.mean.to(self.device)) / self.dataset.std.to(self.device)
                    x_coord_norm_fixed = self._coord_norm(x_fixed_dev)
                    t = torch.rand(B, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                    e = self.W.free_sample(x_fixed_dev).to(self.device)
                    
                    loss_fixed = hilbert_loss_fn(
                        model, self.sde, y_fixed_train_norm, t, e, x_coord_norm_fixed,
                        global_step=step, max_steps=getattr(self, "_max_steps", None),
                        spec_lambda=self.spec_lambda,
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
            sampler = DistributedSampler(self.dataset, shuffle=True,
                                    seed=args.seed if args.seed is not None else 0)
        else:
            sampler = None
        train_loader = data.DataLoader(
            self.dataset,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            sampler=sampler
        )
        steps_per_epoch = len(train_loader)
        self._max_steps = steps_per_epoch * self.config.training.n_epochs
        # Validation loader
        val_loader = data.DataLoader(
            self.test_dataset,
            batch_size=config.training.val_batch_size,
            num_workers=config.data.num_workers,
            shuffle=False
        )
        # Model
        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                        channels=config.model.channels,
                        dim_mults=config.model.dim_mults,
                        is_conditional=config.model.is_conditional)
        elif config.model.model_type == "FNO":
            model = FNO(n_modes=config.model.n_modes, hidden_channels=config.model.hidden_channels, in_channels=config.model.in_channels, out_channels=config.model.out_channels,
                    lifting_channels=config.model.lifting_channels, projection_channels=config.model.projection_channels,
                    n_layers=config.model.n_layers, joint_factorization=config.model.joint_factorization,
                    norm=config.model.norm, preactivation=config.model.preactivation, separable=config.model.separable)
        elif config.model.model_type == "NUFNO":
            from models import NUFNO
            model = NUFNO(config)
        elif config.model.model_type == "AFNO":
            from models.afno import AFNO
            model = AFNO(config)
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],)

        logging.info("Model loaded.")

        # ---------- Optimizer: κ 전용 그룹 + 밴드 게이트 + 시간 게이트 분리 ----------
        base_lr = config.optim.lr
        kappa_lr_mul = float(getattr(config.optim, "kappa_lr_multiplier", 5.0))

        mm_ref = model.module if hasattr(model, "module") else model

        gate_params, time_gate_params, kappa_params, other_params = [], [], [], []

        for n, p in mm_ref.named_parameters():
            if not p.requires_grad:
                continue

            if "kappa_pos_raw" in n:
                kappa_params.append(p)
                continue

            if (".time_gate." in n) or n.endswith("time_gate.tau0") or n.endswith("time_gate.tau1_raw") or n.endswith("time_gate.alpha_raw"):
                time_gate_params.append(p)
                continue

            if ("gate_" in n) and n.endswith("_raw"):
                gate_params.append(p)
                continue

            other_params.append(p)

        param_groups = [
            {"params": other_params, "lr": base_lr, "weight_decay": 0.01},
        ]

        if kappa_params:
            param_groups.append(
                {"params": kappa_params, "lr": base_lr * kappa_lr_mul, "weight_decay": 0.0}
            )

        if gate_params:
            param_groups.append(
                {"params": gate_params, "lr": base_lr, "weight_decay": 0.0}
            )

        if time_gate_params:
            param_groups.append(
                {"params": time_gate_params, "lr": base_lr, "weight_decay": 0.0}
            )

        optimizer = torch.optim.AdamW(param_groups, amsgrad=True)

        # ---------- 정규화 가중치 ----------
        lambda_lf  = float(getattr(config.model, "kappa_lf_lambda", 0.0))
        lambda_ali = float(getattr(config.model, "kappa_aliasing_lambda", 0.0))
        charbonnier_eps = 1e-6

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

                if config.data.dataset == 'Melbourne':
                    y = data_scaler(y)

                t = torch.rand(y.shape[0], device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                # 좌표가 랜덤이면 해당 좌표에 맞춘 Hilbert noise
                if getattr(self.config.data, 'grid_type', 'uniform') in ('random', 'adapted_random'):
                    e = self.W.free_sample(x).to(self.device)
                else:
                    e = self.W.sample(y.shape).to(self.device).squeeze(-1)

                # 스코어 학습 손실 (좌표 가중 포함)
                loss_score = hilbert_loss_fn(
                    model, self.sde, y, t, e, x_coord_norm,
                    global_step=step, max_steps=getattr(self, "_max_steps", None),
                    spec_lambda=self.spec_lambda,
                ).to(self.device)

                # === κ 정규화 항 (평균 스케일) ===
                reg_lf  = y.new_tensor(0.0)
                reg_ali = y.new_tensor(0.0)
                xs = torch.sort(x_coord_norm, dim=1).values
                kappa_max = None
                if xs.size(1) >= 2:
                    dx = (xs[:, 1:] - xs[:, :-1]).clamp_min(1e-8)
                    dx_min_batch = dx.min()
                    kappa_max = (1.0 / dx_min_batch).detach()

                    mm = model.module if hasattr(model, "module") else model
                    if hasattr(mm, "all_kappas") and hasattr(mm, "all_baselines"):
                        for kappa, base in zip(mm.all_kappas(), mm.all_baselines()):
                            k_abs = kappa.to(self.device).abs()
                            b_abs = base.to(self.device).abs()

                            if lambda_lf > 0.0:
                                diff = k_abs - b_abs
                                reg_lf = reg_lf + torch.sqrt(diff.pow(2) + charbonnier_eps**2).mean()

                            if lambda_ali > 0.0:
                                reg_ali = reg_ali + F.softplus(k_abs - kappa_max).mean()

                loss = loss_score
                if lambda_lf  > 0.0: loss = loss + lambda_lf  * reg_lf
                if lambda_ali > 0.0: loss = loss + lambda_ali * reg_ali

                # (기존) Gaussian 데이터셋의 정규화 보조항 유지
                if self.config.data.dataset in ["Gaussian"]:
                    x_raw = x
                    dxs = (x_raw[:, 1:] - x_raw[:, :-1]).abs().to(self.device)
                    t0 = torch.zeros(y.size(0), device=self.device)
                    model_input_t0 = torch.cat([y.unsqueeze(1), x_coord_norm.unsqueeze(1)], dim=1)
                    s_pred = model(model_input_t0, t0)
                    mid = 0.5 * (s_pred[:, 1:] + s_pred[:, :-1]) * dxs
                    log_pdf_pred = torch.cat([torch.zeros(y.size(0), 1, device=self.device), torch.cumsum(mid, dim=1)], dim=1)
                    f = torch.exp(log_pdf_pred)
                    Z_pred = (0.5 * (f[:, 1:] + f[:, :-1]) * dxs).sum(dim=1)
                    norm_loss = (Z_pred - 1.0).pow(2).mean()
                    λ_max = config.training.lambda_norm
                    λ     = λ_max * min(1.0, step / 1000)
                    loss = loss + λ * norm_loss

                # 로깅(분해)
                tb_logger.add_scalar("loss/data", float(loss_score.detach()), step)
                if lambda_ali > 0.0: tb_logger.add_scalar("loss/reg_ali", float(reg_ali.detach()), step)
                if lambda_lf  > 0.0: tb_logger.add_scalar("loss/reg_lf",  float(reg_lf.detach()),  step)
                tb_logger.add_scalar("train_loss", float(torch.abs(loss).detach()), step)

                optimizer.zero_grad()
                loss.backward()

                # ----- Monitoring: gates & kappa (TensorBoard) -----
                if step % 100 == 0:
                    with torch.no_grad():
                        mm = model.module if hasattr(model, "module") else model

                        # 1) 레이어별 게이트(g_low, g_mid, g_high) 로깅
                        if hasattr(mm, "spectral_blocks"):
                            for li, blk in enumerate(mm.spectral_blocks):
                                if hasattr(blk, "gates"):                                
                                    gL, gM, gH = blk.gates()
                                    tb_logger.add_scalar(f"gates/layer{li}_low",  float(gL.detach()), step)
                                    tb_logger.add_scalar(f"gates/layer{li}_mid",  float(gM.detach()), step)
                                    tb_logger.add_scalar(f"gates/layer{li}_high", float(gH.detach()), step)
                                if getattr(blk, "time_gate", None) is not None:
                                    st = blk.time_gate.state()
                                    tb_logger.add_scalar(f"time_gate/layer{li}_tau0",  st["tau0"],  step)
                                    tb_logger.add_scalar(f"time_gate/layer{li}_tau1",  st["tau1"],  step)
                                    tb_logger.add_scalar(f"time_gate/layer{li}_alpha", st["alpha"], step)

                                ks = blk.kappa_for_reg()              # (M,)
                                k_abs = ks.detach().abs()
                                if blk.band_fixed_edges is not None:
                                    th1, th2 = blk.band_fixed_edges
                                    th1 = float(th1)
                                    th2 = float(th2)
                                else:
                                    q1, q2 = blk.band_split_fracs
                                    try:
                                        th1 = torch.quantile(k_abs, q1).item()
                                        th2 = torch.quantile(k_abs, q2).item()
                                    except Exception:
                                        k_sorted, _ = torch.sort(k_abs)
                                        idx1 = int((len(k_sorted) - 1) * q1)
                                        idx2 = int((len(k_sorted) - 1) * q2)
                                        th1 = k_sorted[idx1].item()
                                        th2 = k_sorted[idx2].item()
                                tb_logger.add_scalar(f"gates/layer{li}_th1", th1, step)
                                tb_logger.add_scalar(f"gates/layer{li}_th2", th2, step)

                        # 2) κ 모니터링(절댓값 최대/평균)
                        if hasattr(mm, "all_kappas"):
                            for li, kappa in enumerate(mm.all_kappas()):
                                kappa_now = kappa.detach()
                                tb_logger.add_scalar(f"kappa/layer{li}_abs_max",  float(kappa_now.abs().max()),  step)
                                tb_logger.add_scalar(f"kappa/layer{li}_abs_mean", float(kappa_now.abs().mean()), step)

                        # 3) 배치별 에일리어싱 상한(있을 때) 로깅
                        if kappa_max is not None:
                            tb_logger.add_scalar("debug/kappa_max", float(kappa_max), step)

                        # 4) κ 원시파라미터의 grad-norm 로깅
                        g2 = 0.0
                        for p in kappa_params:
                            if p.grad is not None:
                                g2 += float(p.grad.detach().pow(2).sum().cpu())
                        tb_logger.add_scalar("grad/kappa_pos_raw", g2 ** 0.5, step)

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
                    self.ckpt_dir = os.path.join(args.log_path, f'ckpt_step_{step}.pth')
                    torch.save(model.state_dict(), self.ckpt_dir)
                    latest_ckpt_dir = os.path.join(args.log_path, 'ckpt.pth')
                    torch.save(model.state_dict(), latest_ckpt_dir)

                data_start = time.time()

    def _plot_validation_losses(self, tb_log_dir: str | None = None):
        """
        TensorBoard 로그에서 검증 손실을 읽어와 **개별 그래프**로 시각화하고 저장합니다.
        - 'val_loss/fixed_grid'
        - 'val_loss/resolution_free_...'
        """
        if hasattr(self.config, "tb_logger") and hasattr(self.config.tb_logger, 'log_dir'):
            tb_log_dir = self.config.tb_logger.log_dir
        else:
            tb_log_dir = os.path.join(self.args.exp, "tensorboard", self.args.doc)

        if not os.path.exists(tb_log_dir):
            logging.warning(f"TensorBoard log directory not found: {tb_log_dir}")
            return

        try:
            event_files = [os.path.join(tb_log_dir, f) for f in os.listdir(tb_log_dir) if 'events.out.tfevents' in f]
            if not event_files:
                raise IndexError
            event_file = sorted(event_files, key=os.path.getmtime)[-1]
        except IndexError:
            logging.warning(f"No TensorBoard event file found in {tb_log_dir}")
            return

        logging.info(f"Reading TensorBoard logs from: {event_file}")
        ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        tags = ea.Tags()['scalars']
        
        plot_save_dir = self.args.log_path

        for tag in sorted(tags):
            if tag == 'val_loss/fixed_grid' or tag.startswith('val_loss/resolution_free_'):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                if not steps:
                    continue

                plt.figure(figsize=(10, 6))
                
                if tag == 'val_loss/fixed_grid':
                    plot_label = 'Fixed-Grid Val Loss'
                    plot_title = 'Fixed-Grid Validation Loss over Training'
                    plot_color = 'crimson'
                else:
                    try:
                        points = tag.split('_')[-1]
                        plot_label = f'Res-free ({points} points)'
                        plot_title = f'Resolution-Free Validation Loss ({points} points)'
                        plot_color = 'royalblue'
                    except (IndexError, ValueError):
                        plot_label = tag
                        plot_title = f'Validation Loss for {tag}'
                        plot_color = 'darkslateblue'
                
                plt.plot(steps, values, label=plot_label, color=plot_color)
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.title(plot_title)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)

                filename = tag.replace('/', '_') + '.png'
                save_path = os.path.join(plot_save_dir, f'validation_loss_{filename}')
                
                plt.savefig(save_path)
                plt.close() 
                logging.info(f"Saved validation loss plot to {save_path}")            

    def sample(self, score_model=None):
        args, config = self.args, self.config
        self._plot_validation_losses()
        if config.model.model_type == "ddpm_mnist":
            model = Unet(dim=config.data.image_size,
                         channels=config.model.channels,
                         dim_mults=config.model.dim_mults,
                         is_conditional=config.model.is_conditional,)
        elif config.model.model_type == "FNO":
            model = FNO(n_modes=config.model.n_modes, hidden_channels=config.model.hidden_channels, in_channels=config.model.in_channels, out_channels=config.model.out_channels,
                      lifting_channels=config.model.lifting_channels, projection_channels=config.model.projection_channels,
                      n_layers=config.model.n_layers, joint_factorization=config.model.joint_factorization,
                      norm=config.model.norm, preactivation=config.model.preactivation, separable=config.model.separable)
        elif config.model.model_type == "NUFNO":
            from models import NUFNO
            model = NUFNO(config)            
        elif config.model.model_type == "AFNO":
            from models.afno import AFNO 
            model = AFNO(config)            
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)

        if score_model is not None:
            model = score_model

        elif "ckpt_dir" in config.model.__dict__.keys():
            # Check if specific checkpoint step is requested
            if args.ckpt_step is not None:
                ckpt_path = os.path.join(args.log_path, f'ckpt_step_{args.ckpt_step}.pth')
                if os.path.exists(ckpt_path):
                    ckpt_dir = ckpt_path
                    logging.info(f"Using checkpoint from step {args.ckpt_step}: {ckpt_path}")
                else:
                    logging.warning(f"Checkpoint for step {args.ckpt_step} not found: {ckpt_path}")
                    logging.info("Falling back to latest checkpoint")
                    ckpt_path = os.path.join(args.log_path, 'ckpt.pth')
                    if os.path.exists(ckpt_path):
                        ckpt_dir = ckpt_path
                    else:
                        ckpt_dir = config.model.ckpt_dir
            else:
                # First try the latest checkpoint from training
                ckpt_path = os.path.join(args.log_path, 'ckpt.pth')
                if os.path.exists(ckpt_path):
                    ckpt_dir = ckpt_path
                else:
                    ckpt_dir = config.model.ckpt_dir
            
            states = torch.load(
                ckpt_dir,
                map_location=config.device,
            )

            if args.distributed:
                state_dict = OrderedDict()
                for k, v in states.items():
                    if 'module' in k:
                        name = k[7:]
                        state_dict[name] = v
                    else:
                        state_dict[k] = v

                model.load_state_dict(state_dict)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            else:
                model.load_state_dict(states, strict=False)
        else:
            raise Exception("Fail to load model due to invalid ckpt_dir")

        logging.info("Done loading model")
        model.eval()

        enable_clip = getattr(config.sampling, 'enable_rms_clip', False)
        clip_threshold = getattr(config.sampling, 'rms_clip_threshold', None)
        if enable_clip:
            logging.info(f"RMS clipping enabled with threshold: {clip_threshold}")                  

        test_loader = torch.utils.data.DataLoader(self.test_dataset, config.sampling.batch_size, shuffle=False)
                        
        x_0, y_0 = next(iter(test_loader))
        y_0 = y_0.squeeze(-1)            # (B, N)

        if self.args.disable_resolution_free:
            free_input = x_0.squeeze(-1)
            if config.data.dataset == 'Quadratic':
                y00 = self.test_dataset.inverse_transform(y_0)
            elif config.data.dataset == 'Gaussian':
                y00 = test_loader.dataset.inverse_transform(y_0)
            elif config.data.dataset == 'Doppler' :
                y00 = self.test_dataset.inverse_transform(y_0)   
            else:
                y00 = y_0
        else:
            N_res = self.args.res_free_points[0]             
            if config.data.dataset == 'Quadratic':
                free_input = torch.rand((config.sampling.batch_size, N_res)) * 20 - 10
                free_input = torch.sort(free_input)[0]

                a = torch.randint(low=0, high=2, size=(free_input.shape[0], 1)).repeat(1, N_res) * 2 - 1
                eps = torch.randn(free_input.shape[0], 1).repeat(1, N_res)
                y00 = a * (free_input ** 2) + eps
            elif config.data.dataset == 'Gaussian':
                free_input = torch.rand((config.sampling.batch_size, N_res)) * 20 - 10
                free_input = torch.sort(free_input)[0]
                phi = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * free_input ** 2)
                eps = torch.normal(mean=0., std=0.01, size=(free_input.shape[0], N_res)) * 0.1
                y00 = phi + eps
            elif config.data.dataset == 'Doppler':                    
                N_res = self.args.res_free_points[0]
                free_input = torch.rand((config.sampling.batch_size, N_res), device=self.device).sort(dim=1).values
                noise_std = float(getattr(self.dataset, "noise_std", 0.0))
                b = (torch.randn(config.sampling.batch_size, 1, device=self.device) * noise_std).repeat(1, N_res)
                amp   = torch.sqrt((free_input * (1.0 - free_input)).clamp_min(0.0))
                phase = (2.0 * math.pi * 1.05) / (free_input + 0.05)
                y00 = amp * torch.sin(phase) + b   # 원스케일 GT                

        y_shape = (config.sampling.batch_size, config.data.dimension)      

        if self.args.sample_type in ["srk", "sde"]:        # SRK/Euler 케이스
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
                    method="srk"
                )                 
        elif self.args.sample_type == "tsit5_ode":
            with torch.no_grad():
                t  = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                if self.args.disable_resolution_free:
                    yT = self.W.sample(y_shape).to(self.device)
                else:
                    yT = self.W.free_sample(free_input).to(self.device)
                yT = yT / (torch.std(yT, dim=1, keepdim=True) + 1e-12)
                yT = yT * self.sde.marginal_std(t)[:, None]
                free_input_norm = self._coord_norm(free_input.to(self.device))
                y  = tsit5_sample_ode(
                    model, self.sde,
                    x_t0=yT,
                    x_coord=free_input_norm,
                    device=self.device,
                    inference_steps=self.args.nfe,
                    rtol=1e-5, atol=1e-5,
                    enable_rms_clip=enable_clip,
                    rms_clip_threshold=clip_threshold,                    
                    )

        # ──── Tsit5 결과 시각화 ────
        if self.args.sample_type == "tsit5_ode" and config.data.dataset in ["Quadratic", "Doppler"]:

            x_0   = x_0.cpu()
            y0_plot = self.test_dataset.inverse_transform(y_0).cpu()
            y_plot = (y * self.dataset.std.to(y.device) + self.dataset.mean.to(y.device)).cpu()

            y_pow = y_plot
            y_gt  = y00.cpu()
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[Tsit5] resolution-free power(avg 30) = {power_res}")

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y0_plot[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"Tsit5-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_tsit5.png")
            print("Saved plot fig to visualization_tsit5.png")
            plt.clf(); plt.figure()              

        # ──── Tsit5 결과 : Gaussian ────
        if self.args.sample_type == "tsit5_ode" and config.data.dataset in ["Gaussian"]:

            x_0   = x_0.cpu()
            y_0   = test_loader.dataset.inverse_transform(y_0).cpu()
            y_log = (y * self.dataset.std.to(y.device) + self.dataset.mean.to(y.device))
            y_plot = torch.exp(y_log).cpu()

            y_pow    = y_plot
            y_gt     = y00.cpu()
            n_tests  = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[tsit5_ode] power(avg 30) = {power_res}")

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title("Ground truth (pdf)")

            for i in range(y_plot.shape[0]):
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power={power_res}")

            fig.suptitle(f"tsit5_ode  NFE={self.args.nfe}", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_tsit5.png")
            plt.clf(); plt.figure()                       

        if self.args.sample_type == "srk":
            with torch.no_grad():
                y_shape = (config.sampling.batch_size, config.data.dimension)
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                
            y0_plot = self.test_dataset.inverse_transform(y_0)

            if config.data.dataset == 'Gaussian':
                y_plot = torch.exp(y * self.dataset.std.to(y.device) + self.dataset.mean.to(y.device))
            else:
                y_plot  = y * self.dataset.std.to(y.device) + self.dataset.mean.to(y.device)

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            for i in range(config.sampling.batch_size):
                ax[0].plot(x_0[i, :].cpu(), y0_plot[i, :].cpu())

            ax[0].set_title(f'Ground truth, len:{config.data.hyp_len:.2f}')

            n_tests = config.sampling.batch_size // 10

            for i in range(y.shape[0]):
                ax[1].plot(free_input[i, :].cpu(), y_plot[i, :].cpu(), alpha=1)
            print('Calculate Confidence Interval:')
            power_res = calculate_ci(y_plot, y0_plot, n_tests=n_tests)
            print(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            logging.info(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            ax[1].set_title(f'resolution-free, power(avg of 30 trials): {power_res}')

        else:
            y_0 = y_0.squeeze(-1)
            with torch.no_grad():
                for _ in tqdm(range(1), desc="Generating image samples"):
                    y_shape = (config.sampling.batch_size, config.data.dimension)
                    t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                    y = self.W.sample(y_shape).to(self.device) * self.sde.marginal_std(t)[:, None]
                    y = sampler(y, model, self.sde, self.device, self.W,  self.sde.eps, config.data.dataset)

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            if config.data.dataset == 'Melbourne':
                lp = 10
                n_tests = y.shape[0] // 10
                y = data_inverse_scaler(y)
            if config.data.dataset == 'Gridwatch':
                lp = y.shape[0]
                n_tests = y.shape[0] // 10
                plt.ylim([-2, 3])

            for i in range(lp):
                ax[0].plot(x_0[i, :].cpu(), y[i, :].cpu())
                ax[1].plot(x_0[i, :].cpu(), y_0[i, :].cpu(), c='black', alpha=1)

            ax[0].set_title(f'Ground truth, len:{config.data.hyp_len:.2f}')

            for i in range(lp):
                ax[1].plot(x_0[i, :].cpu(), y[i, :].cpu(), alpha=1)

            power = calculate_ci(y, y_0, n_tests=n_tests)
            print(f'Calculate Confidence Interval: grid, 0th: {power}')

            ax[1].set_title(f'grid, power(avg of 30 trials):{power}')

        # Visualization figure save
        plt.savefig('visualization_default.png')
        print("Saved plot fig to {}".format('visualization_default.png'))
        plt.clf()
        plt.figure()
