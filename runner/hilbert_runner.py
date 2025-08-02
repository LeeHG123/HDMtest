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
from torch.utils.data.distributed import DistributedSampler

from models import *

from functions.utils import *
from functions.loss import hilbert_loss_fn
from functions.sde import VPSDE1D
from functions.sampler import sampler
from functions.unipc_sampler import sample_probability_flow_ode as unipc_sample_ode
from functions.dpmpp_sampler import sample_probability_flow_ode as dpmpp_sample_ode
from functions.deis_sampler  import sample_probability_flow_ode as deis_sample_ode
from functions.tsit5_sampler import sample_probability_flow_ode as tsit5_sample_ode



torch.autograd.set_detect_anomaly(True)

def kernel_se(x1, x2, hyp={'gain':1.0,'len':1.0}):
    """ Squared-exponential kernel function """
    x1_scaled = x1 / hyp['len']
    x2_scaled = x2 / hyp['len']
    D = torch.cdist(x1_scaled, x2_scaled, p=2.0).pow(2) # sqeuclidean
    K = hyp['gain'] * torch.exp(-D)
    return K.to(torch.float32)

class HilbertNoise:
    def __init__(self, grid, x=None, hyp_len=1.0, hyp_gain=1.0, use_truncation=False):
        x_coords = torch.linspace(-10, 10, grid)
        self.hyp = {'gain': hyp_gain, 'len': hyp_len}
        self.x = torch.unsqueeze(x_coords, dim=-1)

        # K, eig_val, eig_vec를 특정 device에 고정하지 않음
        K = kernel_se(self.x, self.x, self.hyp)
        eig_val, eig_vec = torch.linalg.eigh(K + 1e-6 * torch.eye(K.shape[0]))

        self.eig_val = eig_val
        self.eig_vec = eig_vec.to(torch.float32)
        
        # M = E @ sqrt(Λ)
        self.M = self.eig_vec @ torch.diag(torch.sqrt(self.eig_val.clamp(min=0)))
        
        # [핵심 수정] E @ Λ^{-1/2} 를 미리 계산하여 효율성 증대
        self.E_inv_sqrt = self.eig_vec @ torch.diag(1.0 / torch.sqrt(self.eig_val.clamp(min=1e-8)))

    def sample(self, size):
        """고정된 그리드에 대한 노이즈 샘플링"""
        # size: (batch_size, grid_dim)
        device = self.M.device # M이 있는 device를 기준으로 동작
        z = torch.randn(size[0], self.M.shape[1], device=device) 
        output = z @ self.M.T
        return output

    def free_sample(self, free_input_batch):
        device = free_input_batch.device
        batch_size = free_input_batch.shape[0]
        num_points_free = free_input_batch.shape[1]
        
        # 결과를 저장할 텐서
        output_batch = torch.zeros((batch_size, num_points_free), device=device)
        
        # 필요한 텐서들을 free_input_batch와 동일한 device로 이동
        x_fixed = self.x.to(device)
        E_inv_sqrt = self.E_inv_sqrt.to(device) # 미리 계산된 E @ Λ^{-1/2}
        
        # 배치 내 각 샘플에 대해 개별적으로 처리
        for i in range(batch_size):
            current_coords = free_input_batch[i].unsqueeze(-1)
            
            # 1. 크로스-커널 K_yx = K(current_coords, x_fixed) 계산
            K_yx = kernel_se(current_coords, x_fixed, self.hyp)
            
            # 2. 표준 정규분포에서 랜덤 계수 z ~ N(0, I) 샘플링
            z = torch.randn(self.eig_vec.shape[1], 1, device=device)
            
            # 3. 올바른 공식 f_y = K_yx * (E * Λ^{-1/2}) * z 적용
            noise_sample = K_yx @ E_inv_sqrt @ z
            
            output_batch[i] = noise_sample.squeeze()

        return output_batch

class HilbertDiffusion(object):
    def __init__(self, args, config, dataset, test_dataset, device=None):
        self.args = args
        self.config = config
        self.W = HilbertNoise(grid=config.data.dimension, hyp_len=config.data.hyp_len, hyp_gain=config.data.hyp_gain)

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                    else torch.device("cpu")
            )
        self.device = device
        self.num_timesteps = config.diffusion.num_diffusion_timesteps

        self.sde = VPSDE1D(schedule='cosine')

        self.dataset = dataset
        self.test_dataset = test_dataset
        
    def validate(self, model, val_loader, tb_logger, step, calc_fixed_grid_loss: bool = True):
        """
        Validation 함수. Resolution-free 손실과 고정 그리드 손실을 계산합니다.
        
        Args:
            model: 평가할 모델
            val_loader: 검증 데이터 로더 (고정 그리드용)
            tb_logger: TensorBoard 로거
            step: 현재 학습 스텝
            calc_fixed_grid_loss: 고정 그리드 손실을 계산할지 여부 (True/False)
        """
        model.eval()
        
        val_losses_res_free = []
        if calc_fixed_grid_loss:
            val_losses_fixed = []

        with torch.no_grad():
            # val_loader는 배치 크기, shape 등 정보 획득 및 고정 그리드 평가에 사용
            for i, (x_fixed, y_fixed) in enumerate(val_loader):
                if i >= 10:  # 효율성을 위해 검증 배치를 10개로 제한
                    break
                
                B = y_fixed.shape[0]
                N_res = self.args.res_free_points

                if not self.args.disable_resolution_free:
                    # --- 1. Resolution-Free 검증 ---
                    # 랜덤 좌표 생성 ([-10, 10] 범위의 정렬된 랜덤 포인트)
                    x_res_free = (torch.rand(B, N_res, device=self.device) * 20 - 10).sort(dim=1).values

                    # 데이터셋 종류에 따라 Ground Truth y 생성
                    if self.config.data.dataset == 'Quadratic':
                        a = torch.randint(low=0, high=2, size=(B, 1), device=self.device) * 2 - 1
                        eps = torch.randn(B, N_res, device=self.device)
                        y_res_free = a * (x_res_free ** 2) + eps
                        y_res_free = y_res_free / 50.0  # 데이터셋과 동일한 스케일링 적용

                    elif self.config.data.dataset == 'Gaussian':
                        log_phi = -0.5 * x_res_free**2 - 0.5 * math.log(2 * math.pi)
                        eps = torch.randn_like(log_phi) * 1e-2
                        log_phi_noisy = log_phi + eps
                        # 학습 데이터셋의 평균/표준편차로 정규화
                        y_res_free = (log_phi_noisy - self.dataset.mean.to(self.device)) / self.dataset.std.to(self.device)

                    else: # 다른 데이터셋은 현재 로직에서 제외
                        y_res_free = None

                    if y_res_free is not None:
                        x_coord_norm_res_free = x_res_free / 10.0
                        t = torch.rand(B, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                        e = self.W.free_sample(x_res_free).to(self.device)

                        loss_res_free = hilbert_loss_fn(model, self.sde, y_res_free, t, e, x_coord_norm_res_free)
                        val_losses_res_free.append(loss_res_free.item())

                # --- 2. 고정 그리드 검증 (선택적) ---
                if calc_fixed_grid_loss:
                    x_fixed = x_fixed.to(self.device).squeeze(-1)
                    y_fixed = y_fixed.to(self.device).squeeze(-1)
                    x_coord_norm_fixed = x_fixed / 10.0
                    
                    t = torch.rand(B, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                    # 고정 그리드 노이즈는 self.W.sample 사용
                    e = self.W.sample(y_fixed.shape).to(self.device)
                    
                    loss_fixed = hilbert_loss_fn(model, self.sde, y_fixed, t, e, x_coord_norm_fixed)
                    val_losses_fixed.append(loss_fixed.item())

        # --- 결과 계산 및 로깅 ---
        avg_val_loss_res_free = None
        if val_losses_res_free:
            avg_val_loss_res_free = np.mean(val_losses_res_free)
            tb_logger.add_scalar("val_loss/resolution_free", avg_val_loss_res_free, global_step=step)

        avg_val_loss_fixed = None

        if calc_fixed_grid_loss:
            avg_val_loss_fixed = np.mean(val_losses_fixed)
            tb_logger.add_scalar("val_loss/fixed_grid", avg_val_loss_fixed, global_step=step)
        
        model.train()  # 모델을 다시 학습 모드로 설정
        if avg_val_loss_res_free is not None:
            return avg_val_loss_res_free  # Resolution-free 손실을 주된 지표로 반환
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
        elif config.model.model_type == "AFNO":
            from models.afno import AFNO 
            model = AFNO(config)            
        elif config.model.model_type == "ddpm":
            model = Model(config)

        model = model.to(self.device)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],)
                                                            #   find_unused_parameters=True)
        logging.info("Model loaded.")

        # Optimizer, LR scheduler
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)

        # lr_scheduler = get_scheduler(
        #     "linear",
        #     optimizer=optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=2000000,
        # )

        start_epoch, step = 0, 0
        # if args.resume:
        #     states = torch.load(os.path.join(args.log_path, "ckpt.pth"), map_location=self.device)
        #     model.load_state_dict(states[0], strict=False)
        #     start_epoch = states[2]
        #     step = states[3]

        for epoch in range(config.training.n_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device).squeeze(-1)
                y = y.to(self.device).squeeze(-1)
                x_coord_norm = x / 10.0

                data_time += time.time() - data_start
                model.train()
                step += 1

                if config.data.dataset == 'Melbourne':
                    y = data_scaler(y)

                t = torch.rand(y.shape[0], device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                e = self.W.sample(y.shape).to(self.device).squeeze(-1)
 
                loss_score = hilbert_loss_fn(model, self.sde, y, t, e, x_coord_norm).to(self.device)
                # 정규화 보조-손실
                if self.config.data.dataset in ["Gaussian"]:
                    dx = (self.dataset.x[0, 1] - self.dataset.x[0, 0]).abs().to(self.device)

                    # t=0 에서 score(x,0) 추정
                    t0 = torch.zeros(y.size(0), device=self.device)
                    model_input_t0 = torch.cat([y.unsqueeze(1), x_coord_norm.unsqueeze(1)], dim=1) # (B, 2, N) 형태로 만듦
                    score = model(model_input_t0, t0) # 수정된 2채널 입력을 모델에 전달

                    # 누적 적분으로 log p̂(x) 재구성 (C=0 고정)
                    log_pdf_pred = torch.cumsum(score * dx, dim=1)

                    # 적분값 Ẑ ≈ ∑ exp(log p̂) Δx
                    Z_pred = torch.exp(log_pdf_pred).sum(dim=1) * dx

                    # (Ẑ − 1)²
                    norm_loss = (Z_pred - 1.0).pow(2).mean()

                    # λ warm-up (첫 1 K step)
                    λ_max = config.training.lambda_norm      # 1e-2
                    λ     = λ_max * min(1.0, step / 1000)

                    loss = loss_score + λ * norm_loss
                else:
                    loss = loss_score
                tb_logger.add_scalar("train_loss", torch.abs(loss), global_step=step)

                optimizer.zero_grad()
                loss.backward()

                if args.local_rank == 0:
                    logging.info(
                        f"step: {step}, loss: {torch.abs(loss).item()}, data time: {data_time / (i+1)}"
                    )

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass

                optimizer.step()
                # lr_scheduler.step()

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
                    
                    # Also save the latest checkpoint as ckpt.pth
                    latest_ckpt_dir = os.path.join(args.log_path, 'ckpt.pth')
                    torch.save(model.state_dict(), latest_ckpt_dir)

                data_start = time.time()

    def _plot_validation_losses(self, tb_log_dir: str | None = None):
        """
        TensorBoard 로그에서 검증 손실을 읽어와 별도의 그래프로 시각화하고 저장합니다.
        
        Args:
            tb_log_dir: TensorBoard 로그 디렉터리 경로. None이면 자동으로 찾습니다.
        """
        # 1. 실제 SummaryWriter가 사용하는 로그 디렉터리 경로를 확보
        if tb_log_dir is None:
            # 학습 중이라면 tb_logger 객체에서 직접 경로를 가져옴
            if hasattr(self.config, "tb_logger") and hasattr(self.config.tb_logger, 'log_dir'):
                tb_log_dir = self.config.tb_logger.log_dir
                logging.info(f"Plotting: Using log_dir from active tb_logger: {tb_log_dir}")
            # 샘플링 시점이라면 (tb_logger 객체가 없음) 경로를 조합하여 fallback
            else:
                tb_log_dir = os.path.join(self.args.exp, "tensorboard", self.args.doc)
                logging.info(f"Plotting: Using fallback log_dir for sampling: {tb_log_dir}")
        
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
        
        # 그래프 저장 경로
        plot_save_dir = self.args.log_path

        # 2. Resolution-Free Validation Loss 그래프
        tag_res_free = 'val_loss/resolution_free'
        if tag_res_free in tags:
            events = ea.Scalars(tag_res_free)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            if not steps: return # 데이터가 없으면 종료

            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, label='Resolution-Free Val Loss', color='royalblue')
            plt.xlabel('Training Steps'); plt.ylabel('Loss')
            plt.title('Resolution-Free Validation Loss over Training')
            plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
            
            save_path = os.path.join(plot_save_dir, 'validation_loss_resolution_free.png')
            plt.savefig(save_path); plt.close()
            logging.info(f"Saved resolution-free validation loss plot to {save_path}")

        # 3. Fixed-Grid Validation Loss 그래프
        tag_fixed = 'val_loss/fixed_grid'
        if tag_fixed in tags:
            events = ea.Scalars(tag_fixed)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            if not steps: return # 데이터가 없으면 종료

            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, label='Fixed-Grid Val Loss', color='crimson')
            plt.xlabel('Training Steps'); plt.ylabel('Loss')
            plt.title('Fixed-Grid Validation Loss over Training')
            plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

            save_path = os.path.join(plot_save_dir, 'validation_loss_fixed_grid.png')
            plt.savefig(save_path); plt.close()
            logging.info(f"Saved fixed-grid validation loss plot to {save_path}")              

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

        test_loader = torch.utils.data.DataLoader(self.test_dataset, config.sampling.batch_size, shuffle=False)
                        
        # 시각화·평가 루틴이 기대하는 형식 맞추기
        x_0, y_0 = next(iter(test_loader))
        y_0 = y_0.squeeze(-1)            # (B, N)

        if self.args.disable_resolution_free:
            free_input = x_0.squeeze(-1)
            if config.data.dataset == 'Quadratic':
                y00 = y_0 * 50.0
            elif config.data.dataset == 'Gaussian':
                y00 = test_loader.dataset.inverse_transform(y_0)
            else:
                y00 = y_0
        else:
            N_res = self.args.res_free_points            
            if config.data.dataset == 'Quadratic':
                free_input = torch.rand((config.sampling.batch_size, N_res)) * 20 - 10
                free_input = torch.sort(free_input)[0]

                a = torch.randint(low=0, high=2, size=(free_input.shape[0], 1)).repeat(1, N_res) * 2 - 1
                eps = torch.normal(mean=0., std=1., size=(free_input.shape[0], N_res))
                y00 = a * (free_input ** 2) + eps
            elif config.data.dataset == 'Gaussian':
                free_input = torch.rand((config.sampling.batch_size, N_res)) * 20 - 10
                free_input = torch.sort(free_input)[0]
                phi = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * free_input ** 2)
                eps = torch.normal(mean=0., std=0.01, size=(free_input.shape[0], N_res)) * 0.1
                y00 = phi + eps

        y_shape = (config.sampling.batch_size, config.data.dimension)      

        if self.args.sample_type in ["srk", "sde"]:        # SRK/Euler 케이스
            with torch.no_grad():
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                if self.args.disable_resolution_free:
                    y = self.W.sample(y_shape).to(self.device) * self.sde.marginal_std(t)[:, None]
                else:
                    y = self.W.free_sample(free_input).to(self.device) * self.sde.marginal_std(t)[:, None]
                free_input_norm = free_input.to(self.device) / 10.0
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
                 
        elif self.args.sample_type == "unipc_ode":
            with torch.no_grad():
                # SRK 샘플러와 동일하게 초기 시간 T 텐서 생성
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                # 1. HilbertNoise로 상관관계가 있는 노이즈 생성
                if self.args.disable_resolution_free:
                    y_t = self.W.sample(y_shape).to(self.device)
                else:
                    y_t = self.W.free_sample(free_input).to(self.device)

                # 2. 각 샘플(함수)별로 표준편차를 계산
                std_y_t = torch.std(y_t, dim=1, keepdim=True)
                
                # 3. 표준편차로 나누어 스케일을 1로 정규화
                y_t = y_t / (std_y_t)
                
                # 4. SDE의 marginal_std(T)를 곱하여 최종 초기 노이즈 x_T 생성
                y_t = y_t * self.sde.marginal_std(t)[:, None]
                free_input_norm = free_input.to(self.device) / 10.0

                # 생성된 초기 노이즈(y_t)를 x_t0 인자로 전달
                y = unipc_sample_ode(
                    model, self.sde,
                    x_t0=y_t,  # <<< 핵심: Resolution-free 초기 노이즈 전달
                    x_coord=free_input_norm,
                    device=self.device,
                    inference_steps=self.args.nfe,
                )
        elif self.args.sample_type == "dpmpp_ode":
            with torch.no_grad():

                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                if self.args.disable_resolution_free:
                    y_t = self.W.sample(y_shape).to(self.device)
                else:
                    y_t = self.W.free_sample(free_input).to(self.device)
                y_t = y_t / (torch.std(y_t, dim=1, keepdim=True))
                y_t = y_t * self.sde.marginal_std(t)[:, None]
                free_input_norm = free_input.to(self.device) / 10.0
                y = dpmpp_sample_ode(
                    model, self.sde,
                    x_t0=y_t,
                    x_coord=free_input_norm,
                    device=self.device,
                    inference_steps=self.args.nfe,
                )       
        elif self.args.sample_type == "deis_ode":
            with torch.no_grad():
                t  = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                if self.args.disable_resolution_free:
                    yT = self.W.sample(y_shape).to(self.device)
                else:
                    yT = self.W.free_sample(free_input).to(self.device)
                yT = yT / (torch.std(yT, dim=1, keepdim=True))
                yT = yT * self.sde.marginal_std(t)[:, None]
                free_input_norm = free_input.to(self.device) / 10.0
                y  = deis_sample_ode(
                    model, self.sde,
                    x_t0=yT,
                    x_coord=free_input_norm,
                    device=self.device,
                    inference_steps=self.args.nfe,
                    )     
        elif self.args.sample_type == "tsit5_ode":
            with torch.no_grad():
                t  = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                if self.args.disable_resolution_free:
                    yT = self.W.sample(y_shape).to(self.device)
                else:
                    yT = self.W.free_sample(free_input).to(self.device)
                yT = yT / torch.std(yT, dim=1, keepdim=True)
                yT = yT * self.sde.marginal_std(t)[:, None]
                free_input_norm = free_input.to(self.device) / 10.0
                y  = tsit5_sample_ode(
                    model, self.sde,
                    x_t0=yT,
                    x_coord=free_input_norm,
                    device=self.device,
                    inference_steps=self.args.nfe,
                    rtol=1e-5, atol=1e-5,
                    )
         # ──── ①-A UniPC 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "unipc_ode" and config.data.dataset in ["Quadratic"]:

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, config.data.dimension)  균일 그리드
            scale = 50.0 if config.data.dataset == "Quadratic" else (2.0 / math.sqrt(2*math.pi))
            y_0   = (y_0 * scale).cpu()

            # (b) UniPC 생성
            y_plot = (y * scale).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot 
            y_gt  = y00.cpu()                   # (B,res_free_points)
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[UniPC] resolution-free power(avg 30) = {power_res}")

            # (d) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                # SRK와 동일: free_input[i] 에 대응되는 y_plot[i]
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"UniPC-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_unipc.png")
            print("Saved plot fig to visualization_unipc.png")
            plt.clf(); plt.figure()

        # ──── ①-A Dpmpp 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "dpmpp_ode" and config.data.dataset in ["Quadratic"]:

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            scale = 50.0 if config.data.dataset == "Quadratic" else (2.0 / math.sqrt(2*math.pi))
            y_0   = (y_0 * scale).cpu()

            # (b) UniPC 생성
            y_plot = (y * scale).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,res_free_points)
            y_gt  = y00.cpu()                   # (B,res_free_points)
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[Dpmpp] resolution-free power(avg 30) = {power_res}")

            # (d) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                # SRK와 동일: free_input[i] 에 대응되는 y_plot[i]
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"Dpmpp-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_dpmpp.png")
            print("Saved plot fig to visualization_dpmpp.png")
            plt.clf(); plt.figure()           

        # ──── ①-A DEIS 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "deis_ode" and config.data.dataset in ["Quadratic"]:

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, config.data.dimension)  균일 그리드
            scale = 50.0 if config.data.dataset == "Quadratic" else (2.0 / math.sqrt(2*math.pi))
            y_0   = (y_0 * scale).cpu()

            # (b) UniPC 생성
            y_plot = (y * scale).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,res_free_points)
            y_gt  = y00.cpu()                   # (B,res_free_points)
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[DEIS] resolution-free power(avg 30) = {power_res}")

            # (d) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                # SRK와 동일: free_input[i] 에 대응되는 y_plot[i]
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"DEIS-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_deis.png")
            print("Saved plot fig to visualization_deis.png")
            plt.clf(); plt.figure()

        # ──── ①-A Tsit5 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "tsit5_ode" and config.data.dataset in ["Quadratic"]:
            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, config.data.dimension)  균일 그리드
            scale = 50.0 if config.data.dataset == "Quadratic" else (2.0 / math.sqrt(2*math.pi))
            y_0   = (y_0 * scale).cpu()

            # (b) UniPC 생성
            y_plot = (y * scale).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,res_free_points)
            y_gt  = y00.cpu()                   # (B,res_free_points)
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[Tsit5] resolution-free power(avg 30) = {power_res}")

            # (d) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                # SRK와 동일: free_input[i] 에 대응되는 y_plot[i]
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"Tsit5-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_tsit5.png")
            print("Saved plot fig to visualization_tsit5.png")
            plt.clf(); plt.figure()               
                                    
        # ──── ①-A UniPC 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "unipc_ode" and config.data.dataset in ["Gaussian"]:

            # 1) log-값 → pdf 로 복원
            x_0   = x_0.cpu()                                   # (B,config.data.dimension)
            y_0   = test_loader.dataset.inverse_transform(y_0).cpu()
            y_plot = test_loader.dataset.inverse_transform(y).cpu()

            # 2) power 계산
            y_pow = y_plot
            y_gt     = y00.cpu()       # free_input 기반 ground-truth pdf
            n_tests  = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[UniPC-log] power(avg 30) = {power_res}")

            # 3) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title("Ground truth (pdf)")

            for i in range(y_plot.shape[0]):
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power={power_res}")

            fig.suptitle(f"UniPC-ODE  NFE={self.args.nfe}", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_unipc.png")
            plt.clf(); plt.figure()

        # ──── ①-A Dpmpp 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "dpmpp_ode" and config.data.dataset in ["Gaussian"]:

            # 1) log-값 → pdf 로 복원
            x_0   = x_0.cpu()                                   # (B,config.data.dimension)
            y_0   = test_loader.dataset.inverse_transform(y_0).cpu()
            y_plot = test_loader.dataset.inverse_transform(y).cpu()

            # 2) power 계산
            y_pow    = y_plot          # (B,res_free_points)
            y_gt     = y00.cpu()       # free_input 기반 ground-truth pdf
            n_tests  = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[dpmpp_ode] power(avg 30) = {power_res}")

            # 3) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title("Ground truth (pdf)")

            for i in range(y_plot.shape[0]):
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power={power_res}")

            fig.suptitle(f"dpmpp_ode  NFE={self.args.nfe}", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_dpmpp.png")
            plt.clf(); plt.figure()        

        # ──── ①-A DEIS 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "deis_ode" and config.data.dataset in ["Gaussian"]:

            # 1) log-값 → pdf 로 복원
            x_0   = x_0.cpu()                                   # (B,config.data.dimension)
            y_0   = test_loader.dataset.inverse_transform(y_0).cpu()
            y_plot = test_loader.dataset.inverse_transform(y).cpu()

            # 2) power 계산
            y_pow    = y_plot          # (B,res_free_points)
            y_gt     = y00.cpu()       # free_input 기반 ground-truth pdf
            n_tests  = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[deis_ode] power(avg 30) = {power_res}")

            # 3) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title("Ground truth (pdf)")

            for i in range(y_plot.shape[0]):
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power={power_res}")

            fig.suptitle(f"deis_ode  NFE={self.args.nfe}", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_deis.png")
            plt.clf(); plt.figure()  

        # ──── ①-A Tsit5 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "tsit5_ode" and config.data.dataset in ["Gaussian"]:

            # 1) log-값 → pdf 로 복원
            x_0   = x_0.cpu()                                   # (B,config.data.dimension)
            y_0   = test_loader.dataset.inverse_transform(y_0).cpu()
            y_plot = test_loader.dataset.inverse_transform(y).cpu()

            # 2) power 계산
            y_pow    = y_plot          # (B,res_free_points)
            y_gt     = y00.cpu()       # free_input 기반 ground-truth pdf
            n_tests  = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[tsit5_ode] power(avg 30) = {power_res}")

            # 3) 그림
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
                
            scale = 50.0 if config.data.dataset == "Quadratic" \
                           else (2.0 / math.sqrt(2 * math.pi))

            y_0 = y_0 * scale
            y   = y   * scale

            _, ax = plt.subplots(1, 2, figsize=(10, 5))

            for i in range(config.sampling.batch_size):
                ax[0].plot(x_0[i, :].cpu(), y_0[i, :].cpu())

            ax[0].set_title(f'Ground truth, len:{config.data.hyp_len:.2f}')

            n_tests = config.sampling.batch_size // 10

            for i in range(y.shape[0]):
                ax[1].plot(free_input[i, :].cpu(), y[i, :].cpu(), alpha=1)
            print('Calculate Confidence Interval:')
            power_res = calculate_ci(y, y_0, n_tests=n_tests)
            print(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            # power_res2 = calculate_ci(y, y00, n_tests=n_tests)
            # print(f'Calculate Confidence Interval: resolution-free test2, power(avg of 30 trials): {power_res2}')
            logging.info(f'Calculate Confidence Interval: resolution-free, power(avg of 30 trials): {power_res}')
            # logging.info(f'Calculate Confidence Interval: resolution-free test2, power(avg of 30 trials): {power_res2}')
            ax[1].set_title(f'resolution-free, power(avg of 30 trials): {power_res}')
            # ax[1].set_title(f'resfree 1: {power_res}, resfree 2: {power_res2}')
            # plt.savefig('result.png')
            # np.savez(args.log_path + '/rawdata', x_0=x_0.cpu().numpy(), y_0=y_0.cpu().numpy(), free_input=free_input.cpu().numpy(), y=y.cpu().numpy())

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