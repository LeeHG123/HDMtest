import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# FNO 모델의 유틸리티 함수들을 가져옵니다.
from .fno import get_timestep_embedding, Lifting, Projection, default_init, skip_connection

class AFNO1D(nn.Module):
    """
    1D 데이터용 Adaptive Fourier Neural Operator.
    AFNO2D를 1D에 맞게 수정한 버전입니다.
    """
    # --- 수정: __init__에 hard_thresholding_fraction 추가 ---
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hidden_size_factor=2, 
                 hard_thresholding_fraction=1.0, scale=0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hidden_size_factor = hidden_size_factor
        self.scale = scale
        # --- 수정: 하드 임계값 비율 저장 ---
        self.hard_thresholding_fraction = hard_thresholding_fraction

        # 가중치 초기화
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, C, N = x.shape

        # 1. FFT: 1D RFFT 사용으로 변경하여 효율성 증대
        x = torch.fft.rfft(x, n=N, dim=-1, norm="ortho")
        
        N_freqs = x.shape[-1]
        kept_modes = int(N_freqs * self.hard_thresholding_fraction)
        x_low = x[..., :kept_modes] # 저주파수 성분만 선택

        # 채널을 블록으로 나누기 위한 재구성
        x_reshaped = x_low.reshape(B, self.num_blocks, self.block_size, kept_modes)
        x_reshaped = x_reshaped.permute(0, 3, 1, 2) # (B, kept_modes, num_blocks, block_size)

        # 2. BlockMLP in Frequency Domain (저주파수 성분에 대해서만 연산)
        o1_real = F.relu(
            torch.einsum('bnsi,sio->bnso', x_reshaped.real, self.w1[0]) -
            torch.einsum('bnsi,sio->bnso', x_reshaped.imag, self.w1[1]) +
            self.b1[0]
        )
        o1_imag = F.relu(
            torch.einsum('bnsi,sio->bnso', x_reshaped.imag, self.w1[0]) +
            torch.einsum('bnsi,sio->bnso', x_reshaped.real, self.w1[1]) +
            self.b1[1]
        )
        o2_real = (
            torch.einsum('bnsi,sio->bnso', o1_real, self.w2[0]) -
            torch.einsum('bnsi,sio->bnso', o1_imag, self.w2[1]) +
            self.b2[0]
        )
        o2_imag = (
            torch.einsum('bnsi,sio->bnso', o1_imag, self.w2[0]) +
            torch.einsum('bnsi,sio->bnso', o1_real, self.w2[1]) +
            self.b2[1]
        )

        # 3. Soft-shrinkage and Inverse FFT
        x_processed = torch.stack([o2_real, o2_imag], dim=-1)
        x_processed = F.softshrink(x_processed, lambd=self.sparsity_threshold)
        x_processed = torch.view_as_complex(x_processed)
        
        x_processed = x_processed.permute(0, 2, 3, 1) # (B, num_blocks, block_size, kept_modes)
        x_processed = x_processed.reshape(B, C, kept_modes)

        # --- 원래 주파수 텐서 크기로 복원 ---
        x_out = torch.zeros_like(x)
        x_out[..., :kept_modes] = x_processed # 처리된 저주파수 성분만 할당

        x = torch.fft.irfft(x_out, n=N, dim=-1, norm="ortho")
        x = x.type(dtype)

        return x + bias


class AFNO(nn.Module):
    """
    AFNO1D 블록을 여러 층으로 쌓은 전체 모델 래퍼.
    기존 FNO 모델의 구조를 따름.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config.model
        self.n_dim = 1
        self.act = nn.SiLU()

        # 모델 하이퍼파라미터
        self.hidden_channels = self.config.hidden_channels
        self.lifting_channels = self.config.lifting_channels
        self.projection_channels = self.config.projection_channels
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.n_layers = self.config.n_layers

        # 시간 임베딩용 Dense 레이어
        self.Dense = nn.ModuleList([
            nn.Linear(self.lifting_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        ])
        for layer in self.Dense:
            layer.weight.data = default_init()(layer.weight.data.shape)
            nn.init.zeros_(layer.bias)
        
        # 입출력 처리 레이어
        self.lifting = Lifting(in_channels=self.in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim)
        self.projection = Projection(in_channels=self.hidden_channels, out_channels=self.out_channels, hidden_channels=self.projection_channels, n_dim=self.n_dim)

        # AFNO 블록
        self.afno_blocks = nn.ModuleList([
            AFNO1D(
                hidden_size=self.hidden_channels,
                num_blocks=self.config.num_blocks,
                sparsity_threshold=self.config.sparsity_threshold,
                hidden_size_factor=self.config.hidden_size_factor,
                hard_thresholding_fraction=self.config.hard_thresholding_fraction 
            ) for _ in range(self.n_layers)
        ])
        
        # 정규화 및 스킵 연결
        self.norms = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.hidden_channels) for _ in range(self.n_layers)])
        self.skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, n_dim=1, type='soft-gating') for _ in range(self.n_layers)])

    def forward(self, x, t):
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, N) -> (B, 1, N)

        x = self.lifting(x) # (B, 1, N) -> (B, C, N)

        # 시간 임베딩
        temb = get_timestep_embedding(t, self.lifting_channels)
        temb = self.Dense[0](temb)
        temb = self.Dense[1](self.act(temb))
        x = x + temb.unsqueeze(-1) # (B, C, N)

        for i in range(self.n_layers):
            x_skip = self.skips[i](x)
            
            x = self.norms[i](x)
            x = self.afno_blocks[i](x)
            
            x = x + x_skip

        x = self.projection(x)
        x = x.squeeze(1) # (B, 1, N) -> (B, N)
        return x