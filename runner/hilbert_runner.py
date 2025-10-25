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

class _KernelBackboneWrapper(nn.Module):
    """
    Wrap KernelOpHDM so that it matches the score-model interface used by samplers.
    model_input : (B, 2, N) = [signal_channel (= current x_t), coord_norm_channel]
    t          : (B,)
    """
    def __init__(self, hdm_kernel: KernelOpHDM, T: float):
        super().__init__()
        self.hdm_kernel = hdm_kernel
        self.T = float(T) if T is not None else 1.0

    def forward(self, model_input: torch.Tensor, t: torch.Tensor, *, V_batch=None):
        # Unpack channels
        x_t   = model_input[:, 0, :].contiguous()           # (B, N)
        x_nrm = model_input[:, 1, :].contiguous()           # (B, N) in [-1,1]
        x_query = x_nrm.unsqueeze(-1)                       # (B, N, 1)

        # Normalize time for numerical stability
        t_scaled = (t / self.T).to(x_t)

        # x_in_meas: normalized input coords used for measuring x_t
        x_in_meas = x_nrm.unsqueeze(-1)                     # (B, N, 1)

        return self.hdm_kernel(
            U_batch=x_t,
            V_batch=V_batch,
            x_query=x_query,
            t_batch=t_scaled,
            anchors=None,
            x_in_meas=x_in_meas
        )    
   
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
            K_yx = self._K_yx(y)                                   # (Ny, N) float64
            coef = E_inv_sqrt @ z[b].view(-1, 1)                   # (N,1)
            f_y  = K_yx @ coef                                     # (Ny,1)
            out64[b] = f_y.view(-1)

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

        # (기본값) SE 커널 
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

        # --- KNO 적분층 야코비안 보정 상수 주입 ---
        # 좌표 정규화: u = (x - offset) / scale  =>  dx = scale * du
        # measure_scale := scale = dataset.coord_scale
        try:
            ms = float(getattr(dataset, "coord_scale", 1.0))
        except Exception:
            ms = 1.0
        # config.model에 주입해 NUFNO 생성 시 전달되도록 함
        setattr(self.config.model, "measure_scale", ms)          

    def _coord_norm(self, x: torch.Tensor) -> torch.Tensor:
        coord_scale  = getattr(self.dataset, 'coord_scale', 10.0)
        coord_offset = getattr(self.dataset, 'coord_offset', 0.0)
        return (x - coord_offset) / coord_scale   

    @torch.no_grad()
    def _build_kernel_dictionary(
        self,
        x_out_raw: torch.Tensor,
        num_samples: int,
        batch_size: int,
        seed: int = 0,
        *,
        hdm_kernel: KernelOpHDM | None = None,
    ):
        """
        표준 센서 X_out_raw(=X_in_raw) 기준의 사전(U,V) 생성.
        - grid_type == 'uniform': 기존 방식 유지
        - grid_type == 'random' : 매 표본마다 임의 좌표에서 측정한 x_t 를 ψ̃로 X_in에 복원하여 U를 구성
                                  V=-e 는 잠재 z 공유로 표준 센서에 직접 투영(project)
        반환 : U_all ∈ R^{N_dict × (N+1)},  V_all ∈ R^{N_dict × N}
        """
        device = self.device
        torch.manual_seed(seed)
        N = int(x_out_raw.numel())

        # 표준 센서 (학습/사전의 공통 좌표) / 정규화 좌표
        X_canon_raw  = x_out_raw.to(device)                   # (N,)
        X_canon_norm = self._coord_norm(X_canon_raw).unsqueeze(-1)  # (N,1)

        # 데이터 도메인 경계 (임의 격자 생성용)
        xmin = float(getattr(self.dataset, "_xmin", -10.0))
        xmax = float(getattr(self.dataset, "_xmax",  10.0))
        use_random = (str(getattr(self.config.data, "grid_type", "uniform")).lower() == "random")

        U_buf, V_buf = [], []

        for start in range(0, num_samples, batch_size):
            b = min(batch_size, num_samples - start)

            # 1) 시간/스케일
            t   = torch.rand(b, device=device) * (self.sde.T - self.sde.eps) + self.sde.eps
            tau = (t / self.sde.T).unsqueeze(1)              # (b,1)
            a   = self.sde.diffusion_coeff(t).unsqueeze(1)   # (b,1)
            sg  = self.sde.marginal_std(t).unsqueeze(1)      # (b,1)

            # 2) 잠재 z를 뽑고, 같은 z로 임의/표준 격자에 일관 투영
            z_latent = self.W.sample_latent(b)               # (b, m)
            e_canon  = self.W.project(X_canon_raw.unsqueeze(0).expand(b, -1), z_latent).to(device)  # (b,N)

            if not use_random:
                # ── (A) uniform: 과거 파이프라인과 동일 ─────────────────────────────
                X = X_canon_raw.unsqueeze(0).expand(b, -1)                           # (b,N)
                y_raw = self.dataset.generate_raw(X, device=device)                  # (b,N)
                y     = (y_raw - self.dataset.mean.to(device)) / self.dataset.std.to(device)
                # x_t는 표준 센서에서 직접 계산
                x_t_canon = a * y + sg * e_canon                                     # (b,N)
                U = torch.cat([x_t_canon, tau], dim=1)                               # (b,N+1)
                V = -e_canon                                                         # (b,N)

            else:
                # ── (B) random: 매 표본 서로 다른 좌표에서 측정 → ψ̃로 표준 센서 복원 ──
                # 임의 좌표(정렬)와 그 위에서의 깨끗한 함수 / 노이즈
                X_rand_raw  = (torch.rand(b, N, device=device) * (xmax - xmin) + xmin).sort(dim=1).values  # (b,N)
                y_rand_raw  = self.dataset.generate_raw(X_rand_raw, device=device)                         # (b,N)
                y_rand      = (y_rand_raw - self.dataset.mean.to(device)) / self.dataset.std.to(device)

                e_rand = self.W.project(X_rand_raw, z_latent).to(device)                                   # (b,N)
                x_t_rand = a * y_rand + sg * e_rand                                                        # (b,N)

                # ψ̃: 임의 좌표에서 측정한 x_t를 표준 센서(X_in=X_out)로 복원
                #    (inlift는 KernelOpHDM 생성 시 X_in=표준 센서로 초기화됨)
                if (hdm_kernel is not None) and (hdm_kernel.inlift is not None):
                    X_rand_norm = self._coord_norm(X_rand_raw).unsqueeze(-1)                               # (b,N,1)
                    x_t_canon   = hdm_kernel.inlift.lift(X_rand_norm, x_t_rand)                            # (b,N)
                else:
                    # 안전장치: ψ̃가 없으면 y/e를 표준 센서에서 다시 합성(선형성으로 일관성 유지)
                    y_canon_raw = self.dataset.generate_raw(X_canon_raw.unsqueeze(0).expand(b, -1), device=device)
                    y_canon     = (y_canon_raw - self.dataset.mean.to(device)) / self.dataset.std.to(device)
                    x_t_canon   = a * y_canon + sg * e_canon

                U = torch.cat([x_t_canon, tau], dim=1)     # (b, N+1)
                V = -e_canon                                # (b, N)

            U_buf.append(U.cpu())
            V_buf.append(V.cpu())

        U_all = torch.cat(U_buf, dim=0)
        V_all = torch.cat(V_buf, dim=0)
        return U_all, V_all                   
        
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
                        if str(self.config.data.dataset).lower() == 'gaussian':
                            xmin, xmax = -10.0, 10.0
                        else:
                            xmin = float(getattr(self.dataset, "_xmin", -10.0))
                            xmax = float(getattr(self.dataset, "_xmax",  10.0))

                        x_res_free = (torch.rand(B, N_res, device=self.device) * (xmax - xmin) + xmin).sort(dim=1).values

                        # GT 생성
                        if str(self.config.data.dataset).lower() == 'gaussian':
                            # 기존 Gaussian 분기 유지: log φ(x) + 작은 잡음 → (학습셋 통계) 정규화
                            log_phi = -0.5 * x_res_free**2 - 0.5 * math.log(2 * math.pi)
                            eps = torch.randn_like(log_phi) * 1e-2
                            log_phi_noisy = log_phi + eps
                            y_res_free = (log_phi_noisy - self.dataset.mean.to(self.device)) / self.dataset.std.to(self.device)
                        else:
                            # 모든 비 Gaussian(Quadratic/Linear/Sin/Circle/Doppler)은 generate_raw로 통일
                            y_raw = self.dataset.generate_raw(x_res_free, device=self.device)
                            y_res_free = (y_raw - self.dataset.mean.to(self.device)) / self.dataset.std.to(self.device)

                        # 손실 계산 (좌표 정규화 포함)
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
        elif config.model.model_type == "KERNEL":
            logging.info("[KERNEL] No learnable parameters. Skipping training and moving to sampling.")
            return                           

        model = model.to(self.device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],)

        logging.info("Model loaded.")

        # ---------- Optimizer: κ 전용 그룹 + 밴드 게이트 + 시간 게이트 분리 ----------
        base_lr = config.optim.lr
        kappa_lr_mul = float(getattr(config.optim, "kappa_lr_multiplier", 5.0))

        mm_ref = model.module if hasattr(model, "module") else model

        gate_params, time_gate_params, kappa_params, other_params = [], [], [], []

        # KNO 커널 파라미터 그룹
        kno_amp_params   = []   # log_gain (RBF), log_w (GSM)
        kno_bw_params    = []   # log_len (RBF), log_sig (GSM)
        kno_freq_params  = []   # log_mu (GSM)  — NS-GSM은 feat 네트로 대체
        kno_feat_params  = []   # NS-GSM: kern.feat.* (좌표→w,σ,μ 생성 NN)
        kno_tcond_params = []   # kern.tmlp.* (시간조건 게이트)        

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

            # KNO 경로
            if ".impl.kern." in n:
                if n.endswith(".log_gain") or n.endswith(".log_w"):
                    kno_amp_params.append(p); continue
                if n.endswith(".log_len") or n.endswith(".log_sig"):
                    kno_bw_params.append(p); continue
                if n.endswith(".log_mu"):
                    kno_freq_params.append(p); continue
                if ".kern.feat." in n:   # NS-GSM의 좌표-의존 파라미터 네트워크
                    kno_feat_params.append(p); continue
                if ".kern.tmlp." in n:   # 시간조건 모듈
                    kno_tcond_params.append(p); continue            

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

        # KNO 
        if kno_amp_params:
            param_groups.append({"params": kno_amp_params,  "lr": base_lr * 1.00, "weight_decay": 1e-4})
        if kno_bw_params:
            param_groups.append({"params": kno_bw_params,   "lr": base_lr * 0.50, "weight_decay": 0.0})
        if kno_freq_params:
            param_groups.append({"params": kno_freq_params, "lr": base_lr * 0.25, "weight_decay": 0.0})
        if kno_feat_params:
            param_groups.append({"params": kno_feat_params, "lr": base_lr * 0.50, "weight_decay": 0.0})
        if kno_tcond_params:
            param_groups.append({"params": kno_tcond_params,"lr": base_lr * 0.50, "weight_decay": 0.0})            

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
                                # ── NUDFT 전용 κ/밴드 로깅: 존재할 때만 실행 ──
                                if hasattr(blk, "kappa_for_reg"):
                                    ks = blk.kappa_for_reg().detach().abs()
                                    q1, q2 = getattr(blk, "band_split_fracs", (0.4, 0.8))
                                    try:
                                        th1 = torch.quantile(ks, q1).item()
                                        th2 = torch.quantile(ks, q2).item()
                                    except Exception:
                                        k_sorted, _ = torch.sort(ks)
                                        i1 = int((len(k_sorted) - 1) * q1)
                                        i2 = int((len(k_sorted) - 1) * q2)
                                        th1, th2 = k_sorted[i1].item(), k_sorted[i2].item()
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
        elif config.model.model_type == "KERNEL":
            # 1) 센서 좌표(정규화)
            x_fixed, _ = next(iter(torch.utils.data.DataLoader(self.test_dataset, 1)))
            x_fixed = x_fixed.squeeze(0).squeeze(-1).to(self.device)         # (N,)
            X_out_norm = self._coord_norm(x_fixed).unsqueeze(-1)             # (N,1)
            X_in_norm  = X_out_norm.clone()

            kcfg = getattr(config.model, "kernel", None)
            def _get(ns, key, default=None):
                return getattr(ns, key) if (ns is not None and hasattr(ns, key)) else default

            # 2) 상위 옵션
            in_kernel   = _get(kcfg, "in_kernel",  "rq")
            out_kernel  = _get(kcfg, "out_kernel", "m52")
            solver      = _get(kcfg, "solver",     "exact")
            use_time    = bool(_get(kcfg, "use_time", True))
            use_prod    = bool(_get(kcfg, "use_product_time", True))
            time_kernel = _get(kcfg, "time_kernel", "rbf")
            nys_m       = _get(kcfg, "nystrom_m", None)  # 힌트용(현재 포워드에서 미설정 시 배치로부터 자동 선택)

            in_init   = _get(kcfg, "in_init", None)
            out_init  = _get(kcfg, "out_init", None)
            time_init = _get(kcfg, "time_init", None)
            inlift    = _get(kcfg, "inlift", None)

            # 3) KERNEL 백본 생성(초기 하이퍼/ψ̃ 옵션 전달)
            hdm_kernel = KernelOpHDM(
                X_in=X_in_norm,
                X_out=X_out_norm,
                in_kernel=in_kernel,
                out_kernel=out_kernel,
                solver=solver,
                nystrom_anchors=None,
                use_time=use_time,
                device=self.device,
                use_product_time=use_prod,
                time_kernel=time_kernel,
                # --- 초기 하이퍼 ---
                in_init_lengthscale=_get(in_init, "lengthscale", None),
                in_init_variance=float(_get(in_init, "variance", 1.0)) if in_init else 1.0,
                in_init_lambda=float(_get(in_init, "lambda", 1e-6))   if in_init else 1e-6,
                time_init_lengthscale=_get(time_init, "lengthscale", None),
                time_init_variance=float(_get(time_init, "variance", 1.0)) if time_init else 1.0,
                out_init_lengthscale=_get(out_init, "lengthscale", None),
                out_init_variance=float(_get(out_init, "variance", 1.0)) if out_init else 1.0,
                out_init_rho=float(_get(out_init, "rho", 1e-6)) if out_init else 1e-6,
                # --- ψ̃(inlift) ---
                inlift_enable=bool(_get(inlift, "enable", True)) if inlift else True,
                inlift_kernel=_get(inlift, "kernel", "m52") if inlift else "m52",
                inlift_init_lengthscale=_get(inlift, "lengthscale", None),
                inlift_init_variance=float(_get(inlift, "variance", 1.0)) if inlift else 1.0,
                inlift_rho=float(_get(inlift, "rho", 1e-8)) if inlift else 1e-8,
            )

            # 4) 사전(U,V) 로드/생성
            dcfg = _get(kcfg, "dict", None)
            dict_size  = int(_get(dcfg, "size",  getattr(config.model, "kernel_dict_size", 4096)))
            dict_batch = int(_get(dcfg, "batch", getattr(config.model, "kernel_dict_batch", 256)))
            dict_path  = _get(dcfg, "path",  os.path.join(self.args.log_path, "kernel_dict.pt"))

            try:
                data = torch.load(dict_path, map_location=self.device)
                U_all = data["U"].to(self.device)
                V_all = data["V"].to(self.device)
                print(f"[KERNEL] Loaded dictionary from: {dict_path} (|U|={U_all.size(0)})")
            except Exception:
                N_dict = dict_size
                B_dict = dict_batch
                seed   = self.args.seed if getattr(self.args, "seed", None) is not None else 0
                # >>> 변경: hdm_kernel을 넘겨 ψ̃ 복원을 사용
                U_all, V_all = self._build_kernel_dictionary(
                    x_out_raw=x_fixed,
                    num_samples=N_dict,
                    batch_size=B_dict,
                    seed=seed,
                    hdm_kernel=hdm_kernel,
                )
                os.makedirs(self.args.log_path, exist_ok=True)
                torch.save({"U": U_all.cpu(), "V": V_all.cpu()}, dict_path)
                print(f"[KERNEL] Built and saved dictionary to: {dict_path} (|U|={U_all.size(0)})")

            # 5) 하이퍼 MLE (옵션)
            mle_cfg  = _get(kcfg, "mle", None)
            in_mle   = _get(mle_cfg, "input",  None)
            out_mle  = _get(mle_cfg, "output", None)

            if _get(in_mle, "enable", True):
                hdm_kernel.fit_input_kernel_hyperparams(
                    U_all, V_all,
                    max_iter=int(_get(in_mle, "max_iter", 120)),
                    lr=float(_get(in_mle, "lr", 0.2)),
                    optimize_alpha=bool(_get(in_mle, "optimize_alpha", True)),
                    optimize_time=bool(_get(in_mle, "optimize_time", True)),
                )
            if _get(out_mle, "enable", True):
                hdm_kernel.fit_output_kernel_hyperparams(
                    V_all,
                    max_iter=int(_get(out_mle, "max_iter", 100)),
                    lr=float(_get(out_mle, "lr", 0.2)),
                    optimize_alpha=bool(_get(out_mle, "optimize_alpha", False)),
                )

            # 6) 사전 등록 후 스코어 모델 래핑
            hdm_kernel.register_dictionary(U_all, V_all)
            # ===== DEBUG: one-shot regression of -e (no ODE) =====
            with torch.no_grad():
                B = 64
                N = x_fixed.numel()
                t  = torch.rand(B, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                a  = self.sde.diffusion_coeff(t).unsqueeze(1)
                sg = self.sde.marginal_std(t).unsqueeze(1)

                Xrep  = x_fixed.unsqueeze(0).expand(B, -1)  # (B,N)
                y_raw = self.dataset.generate_raw(Xrep, device=self.device)
                y     = (y_raw - self.dataset.mean.to(self.device)) / self.dataset.std.to(self.device)

                z_latent = self.W.sample_latent(B)
                e        = self.W.project(Xrep, z_latent).to(self.device)   # (B,N)

                x_t  = a * y + sg * e                                       # (B,N)
                tau  = (t / self.sde.T).unsqueeze(1)                        # (B,1)
                U_q  = torch.cat([x_t, tau], dim=1)                         # (B,N+1)

                x_coord_norm = self._coord_norm(Xrep)
                x_query      = x_coord_norm.unsqueeze(-1)                   # (B,N,1)

                y_hat = hdm_kernel.query(U_q, x_query)                      # (B,N)  ← 예측 -e

                # trapz-가중 MSE
                def _trapz_w(x):
                    xs,_ = torch.sort(x, dim=1); dx = xs[:,1:] - xs[:,:-1]
                    w = torch.zeros_like(x); w[:,0]=0.5*dx[:,0]; w[:,-1]=0.5*dx[:,-1]
                    if x.size(1)>2: w[:,1:-1]=0.5*(dx[:,1:]+dx[:,:-1])
                    return w/(w.sum(dim=1,keepdim=True)+1e-12)
                w   = _trapz_w(x_coord_norm)
                mse = ((y_hat + e)**2 * w).sum(dim=1).mean().item()
                print(f"[KERNEL DEBUG] holdout MSE for predicting -e: {mse:.6e}")            
            model = _KernelBackboneWrapper(hdm_kernel, T=self.sde.T).to(self.device)                                                    
          
        model = model.to(self.device)

        is_kernel = (self.config.model.model_type == "KERNEL")

        if score_model is not None:
            model = score_model  

        elif (not is_kernel) and ("ckpt_dir" in config.model.__dict__.keys()):
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

            if os.path.exists(ckpt_dir):
                states = torch.load(ckpt_dir, map_location=config.device)
                if args.distributed:
                    state_dict = OrderedDict()
                    for k, v in states.items():
                        name = k[7:] if k.startswith('module.') else k
                        state_dict[name] = v
                    model.load_state_dict(state_dict, strict=False)
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
                else:
                    model.load_state_dict(states, strict=False)
            else:
                logging.warning(f"Checkpoint not found: {ckpt_dir} — skipping load (model_type={self.config.model.model_type}).")

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
            free_input = x_0.squeeze(-1)  # 고정 그리드
            if str(config.data.dataset).lower() in ('quadratic', 'linear', 'circle', 'sin', 'sinc', 'doppler', 'rq', 'matern12', 'matern32', 'matern52', 'blocks'):
                y00 = self.test_dataset.inverse_transform(y_0)
            elif str(config.data.dataset).lower() == 'gaussian':
                y00 = test_loader.dataset.inverse_transform(y_0)
            else:
                y00 = y_0  # fallback
        else:
            N_res = self.args.res_free_points[0]
            name = str(config.data.dataset).lower()
            if name == 'gaussian':
                # 기존 Gaussian 경로 유지
                free_input = torch.rand((config.sampling.batch_size, N_res)) * 20 - 10
                free_input = torch.sort(free_input)[0].to(self.device)
                phi = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * free_input ** 2)
                eps = torch.normal(mean=0., std=0.01, size=(free_input.shape[0], N_res), device=self.device) * 0.1
                y00 = phi + eps  # 원스케일 GT(pdf)
            else:
                # 모든 비 Gaussian(Quadratic/Linear/Sin/Circle/Doppler)을 dataset.generate_raw로 통일
                xmin = float(getattr(self.dataset, "_xmin", -10.0))
                xmax = float(getattr(self.dataset, "_xmax",  10.0))
                free_input = (torch.rand((config.sampling.batch_size, N_res), device=self.device) * (xmax - xmin) + xmin)
                free_input = torch.sort(free_input, dim=1)[0]
                y00 = self.dataset.generate_raw(free_input, device=self.device)  # 원스케일 GT          

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
                    # pass Hilbert-noise kernel & coord info
                    hilbert_len=float(self.config.data.hyp_len),
                    hilbert_gain=float(self.config.data.hyp_gain),
                    hilbert_metric="euclidean",               # utils.kernel과 일치
                    coord_scale=float(self.dataset.coord_scale),
                    coord_offset=float(self.dataset.coord_offset),
                    cov_jitter=1e-6,                                      
                    )

        # ──── Tsit5 결과 시각화 ────
        if self.args.sample_type == "tsit5_ode" and config.data.dataset in ["Quadratic", "Linear", "Circle", "Sin", "Sinc", "Doppler", "RQ", "Matern12", "Matern32", "Matern52", "Blocks"]:
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
