import os
import logging
from scipy.spatial import distance
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt

from evaluate.power import calculate_ci
from datasets import data_scaler, data_inverse_scaler

from collections import OrderedDict

import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

from models import *

from functions.utils import *
from functions.loss import hilbert_loss_fn, hilbert_sobolev_loss_fn
from functions.sde import VPSDE1D
from functions.sampler import sampler
from functions.unipc_sampler import sample_probability_flow_ode as unipc_sample_ode
from functions.dpmpp_sampler import sample_probability_flow_ode as dpmpp_sample_ode
from functions.deis_sampler  import sample_probability_flow_ode as deis_sample_ode
from functions.tsit5_sampler import sample_probability_flow_ode as tsit5_sample_ode
from functions.pfdiff_sampler import sample_probability_flow_ode as pfdiff_sample_ode



torch.autograd.set_detect_anomaly(True)

def kernel_se(x1, x2, hyp={'gain':1.0,'len':1.0}):
    """ Squared-exponential kernel function """
    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()

    D = distance.cdist(x1/hyp['len'],x2/hyp['len'],'sqeuclidean')
    K = hyp['gain']*np.exp(-D)
    return torch.from_numpy(K).to(torch.float32)

class HilbertNoise:
    def __init__(self, grid, x=None, hyp_len=1.0, hyp_gain=1.0, use_truncation=False):
        x = torch.linspace(-10, 10, grid)
        self.hyp = {'gain': hyp_gain, 'len': hyp_len}
        x = torch.unsqueeze(x, dim=-1)
        self.x = x
        if x is not None:
            self.x=x

        K = kernel_se(x, x, self.hyp)
        K = K.cpu().numpy()
        eig_val, eig_vec = np.linalg.eigh(K + 1e-6 * np.eye(K.shape[0], K.shape[0]))

        self.eig_val = torch.from_numpy(eig_val)
        self.eig_vec = torch.from_numpy(eig_vec).to(torch.float32)
        self.D = torch.diag(self.eig_val).to(torch.float32)
        self.M = torch.matmul(self.eig_vec, torch.sqrt(self.D))

    def sample(self, size):
        size = list(size)  # batch*grid
        x_0 = torch.randn(size)

        output = (x_0 @ self.M.transpose(0, 1))  # batch grid x grid x grid
        return output  # bath*grid

    def free_sample(self, free_input):  # input (batch,grid)

        y = torch.randn(len(free_input), self.x.shape[0]) @ self.eig_vec.T @ kernel_se(self.x, free_input[0].unsqueeze(-1), self.hyp)
        return y

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
        
    def validate(self, model, val_loader, tb_logger, step):
        """Validation function to compute validation loss"""
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= 10:  # Limit validation to 10 batches for efficiency
                    break
                    
                x = x.to(self.device).squeeze(-1)
                y = y.to(self.device).squeeze(-1)
                
                if self.config.data.dataset == 'Melbourne':
                    y = data_scaler(y)
                
                t = torch.rand(y.shape[0], device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                e = self.W.sample(y.shape).to(self.device).squeeze(-1)
                
                val_loss = hilbert_sobolev_loss_fn(model, self.sde, y, t, e, sobolev_weight=self.config.training.sobolev_weight).to(self.device)
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        tb_logger.add_scalar("val_loss", avg_val_loss, global_step=step)
        
        model.train()  # Set back to training mode
        return avg_val_loss

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

                data_time += time.time() - data_start
                model.train()
                step += 1

                if config.data.dataset == 'Melbourne':
                    y = data_scaler(y)

                t = torch.rand(y.shape[0], device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps
                e = self.W.sample(y.shape).to(self.device).squeeze(-1)
 
                loss = hilbert_sobolev_loss_fn(model, self.sde, y, t, e, sobolev_weight=config.training.sobolev_weight).to(self.device)
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
                if step % config.training.val_freq == 0:
                    val_loss = self.validate(model, val_loader, tb_logger, step)
                    if args.local_rank == 0:
                        logging.info(f"step: {step}, val_loss: {val_loss}")

                if step % config.training.ckpt_store == 0:
                    self.ckpt_dir = os.path.join(args.log_path, f'ckpt_step_{step}.pth')
                    torch.save(model.state_dict(), self.ckpt_dir)
                    
                    # Also save the latest checkpoint as ckpt.pth
                    latest_ckpt_dir = os.path.join(args.log_path, 'ckpt.pth')
                    torch.save(model.state_dict(), latest_ckpt_dir)

                data_start = time.time()

    def sample(self, score_model=None):
        args, config = self.args, self.config

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

        if config.data.dataset == 'Quadratic':
            free_input = torch.rand((config.sampling.batch_size, y_0.shape[1])) * 20 - 10
            free_input = torch.sort(free_input)[0]

            a = torch.randint(low=0, high=2, size=(free_input.shape[0], 1)).repeat(1, 100) * 2 - 1
            eps = torch.normal(mean=0., std=1., size=(free_input.shape[0], 1)).repeat(1, 100)
            y00 = a * (free_input ** 2) + eps

        if self.args.sample_type in ["srk", "sde"]:        # SRK/Euler 케이스
            with torch.no_grad():
                y_shape = (config.sampling.batch_size, config.data.dimension)
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                y = self.W.free_sample(free_input).to(self.device) * self.sde.marginal_std(t)[:, None]
                y = sampler(
                    y, model, self.sde, self.device, self.W,
                    self.sde.eps, config.data.dataset,
                    steps=self.args.nfe,
                    method="srk")     
        elif self.args.sample_type == "unipc_ode":
            with torch.no_grad():
                # SRK 샘플러와 동일하게 초기 시간 T 텐서 생성
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T

                # 1. HilbertNoise로 상관관계가 있는 노이즈 생성
                y_t = self.W.free_sample(free_input).to(self.device)

                # 2. 각 샘플(함수)별로 표준편차를 계산
                std_y_t = torch.std(y_t, dim=1, keepdim=True)
                
                # 3. 표준편차로 나누어 스케일을 1로 정규화
                y_t = y_t / (std_y_t)
                
                # 4. SDE의 marginal_std(T)를 곱하여 최종 초기 노이즈 x_T 생성
                y_t = y_t * self.sde.marginal_std(t)[:, None]

                # 생성된 초기 노이즈(y_t)를 x_t0 인자로 전달
                y = unipc_sample_ode(
                    model, self.sde,
                    x_t0=y_t,  # <<< 핵심: Resolution-free 초기 노이즈 전달
                    device=self.device,
                    inference_steps=self.args.nfe,
                )
        elif self.args.sample_type == "pfdiff_ode":
            with torch.no_grad():
                t  = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                yT = self.W.free_sample(free_input).to(self.device)
                yT = yT / torch.std(yT, dim=1, keepdim=True)
                yT = yT * self.sde.marginal_std(t)[:, None]

                y  = pfdiff_sample_ode(
                    model, self.sde,
                    x_t0=yT,
                    device=self.device,
                    inference_steps=self.args.nfe,
                )               
        elif self.args.sample_type == "dpmpp_ode":
            with torch.no_grad():

                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                y_t = self.W.free_sample(free_input).to(self.device)
                y_t = y_t / (torch.std(y_t, dim=1, keepdim=True))
                y_t = y_t * self.sde.marginal_std(t)[:, None]

                y = dpmpp_sample_ode(
                    model, self.sde,
                    x_t0=y_t,
                    device=self.device,
                    inference_steps=self.args.nfe,
                )       
        elif self.args.sample_type == "deis_ode":
            with torch.no_grad():
                t  = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                yT = self.W.free_sample(free_input).to(self.device)
                yT = yT / (torch.std(yT, dim=1, keepdim=True))
                yT = yT * self.sde.marginal_std(t)[:, None]

                y  = deis_sample_ode(
                    model, self.sde,
                    x_t0=yT,
                    device=self.device,
                    inference_steps=self.args.nfe,
                    )     
        elif self.args.sample_type == "tsit5_ode":
            with torch.no_grad():
                t  = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                yT = self.W.free_sample(free_input).to(self.device)
                yT = yT / torch.std(yT, dim=1, keepdim=True)
                yT = yT * self.sde.marginal_std(t)[:, None]
                y  = tsit5_sample_ode(
                    model, self.sde,
                    x_t0=yT,
                    device=self.device,
                    inference_steps=self.args.nfe,
                    rtol=1e-5, atol=1e-5,
                    )
                                    
        # ──── ①-A UniPC 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "unipc_ode" and config.data.dataset == "Quadratic":
            import matplotlib.pyplot as plt

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            y_0   = (y_0 * 50).cpu()     # GT 스케일-복원

            # (b) UniPC 생성
            y_plot = (y * 50).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,100)
            y_gt  = y00.cpu()                   # (B,100)
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
        # ──── ①-A PFDiff 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "pfdiff_ode" and config.data.dataset == "Quadratic":
            import matplotlib.pyplot as plt

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            y_0   = (y_0 * 50).cpu()     # GT 스케일-복원

            # (b) UniPC 생성
            y_plot = (y * 50).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,100)
            y_gt  = y00.cpu()                   # (B,100)
            n_tests   = y_pow.shape[0] // 10
            power_res = calculate_ci(y_pow, y_gt, n_tests=n_tests)
            print(f"[PFDiff] resolution-free power(avg 30) = {power_res}")

            # (d) 그림
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(10):
                ax[0].plot(x_0[i], y_0[i], color="k", alpha=.7)
            ax[0].set_title(f"Ground truth, len:{config.data.hyp_len:.2f}")

            for i in range(y_plot.shape[0]):
                # SRK와 동일: free_input[i] 에 대응되는 y_plot[i]
                ax[1].plot(free_input[i].cpu(), y_plot[i], alpha=.9)
            ax[1].set_title(f"resolution-free, power(avg 30): {power_res}")

            fig.suptitle(f"PFDiff-ODE (NFE={self.args.nfe})", fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization_pfdiff.png")
            print("Saved plot fig to visualization_pfdiff.png")
            plt.clf(); plt.figure()            

        # ──── ①-A Dpmpp 결과 시각화 : SRK 스타일(2-패널) ────
        if self.args.sample_type == "dpmpp_ode" and config.data.dataset == "Quadratic":
            import matplotlib.pyplot as plt

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            y_0   = (y_0 * 50).cpu()     # GT 스케일-복원

            # (b) UniPC 생성
            y_plot = (y * 50).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,100)
            y_gt  = y00.cpu()                   # (B,100)
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
        if self.args.sample_type == "deis_ode" and config.data.dataset == "Quadratic":
            import matplotlib.pyplot as plt

            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            y_0   = (y_0 * 50).cpu()     # GT 스케일-복원

            # (b) UniPC 생성
            y_plot = (y * 50).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,100)
            y_gt  = y00.cpu()                   # (B,100)
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
        if self.args.sample_type == "tsit5_ode" and config.data.dataset == "Quadratic":
            import matplotlib.pyplot as plt
            # (a) 기존 SRK 코드와 동일하게 x_0, y_0 사용
            x_0   = x_0.cpu()            # (B, 100)  균일 그리드
            y_0   = (y_0 * 50).cpu()     # GT 스케일-복원

            # (b) UniPC 생성
            y_plot = (y * 50).cpu()

            # (c) power 평가 (resolution-free와 동일 로직)
            y_pow = y_plot                # (B,100)
            y_gt  = y00.cpu()                   # (B,100)
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

        elif self.args.sample_type == "srk":
            with torch.no_grad():
                y_shape = (config.sampling.batch_size, config.data.dimension)
                t = torch.ones(config.sampling.batch_size, device=self.device) * self.sde.T
                
            y_0 = y_0 * 50
            y = y * 50

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