import os

import torch
from torch import nn, einsum
import tqdm
import numpy as np
from einops import rearrange

from tqdm.asyncio import trange, tqdm

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from scipy.spatial.distance import cdist

def kernel(x, y, gain=1.0, lens=1, metric='seuclidean', device='cuda'):
    """Compute kernel function for 1D data."""
    x = x.cpu()
    y = y.cpu()
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    
    # Compute distance matrix
    dist = cdist(x, y, metric=metric)
    K = torch.from_numpy(dist / lens)
    K = gain * torch.exp(-K).to(torch.float32)
    
    return K.to(device).to(torch.float32)


class hilbert_noise:
    """Hilbert space noise generation for 1D data."""
    def __init__(self, config, device='cuda'):
        self.grid = grid = config.diffusion.grid
        self.device = device
        self.metric = metric = config.diffusion.metric 
        self.initial_point = config.diffusion.initial_point
        self.end_point = config.diffusion.end_point

        self.x = torch.linspace(config.diffusion.initial_point, config.diffusion.end_point, grid).to(self.device)
        
        self.lens = lens = config.diffusion.lens
        self.gain = gain = config.diffusion.gain
        
        # Compute kernel matrix for 1D data
        K = kernel(self.x, self.x, lens=lens, gain=gain, metric=metric, device=device)
        
        # Eigendecomposition
        eig_val, eig_vec = torch.linalg.eigh(K + 1e-6 * torch.eye(K.shape[0], K.shape[0]).to(self.device))
        self.eig_val = eig_val.to(self.device) 
        self.eig_vec = eig_vec.to(torch.float32).to(self.device) 
        print('eig_val', eig_val.min(), eig_val.max())
        self.D = torch.diag(self.eig_val).to(torch.float32).to(self.device) 
        self.M = torch.matmul(self.eig_vec, torch.sqrt(self.D)).to(self.device)
 
    def sample(self, size):
        """Generate samples in Hilbert space for 1D data."""
        size = list(size)  # batch*ch*grid
        x_0 = torch.randn(size).to(self.device)  

        # Transform to Hilbert space
        output = einsum('g k, b c k -> b c g', self.M, x_0)

        return output  # (batch, ch, grid)

    def free_sample(self, resolution_grid):
        """Generate samples at different resolution for 1D data."""
        y = torch.linspace(self.initial_point, self.end_point, resolution_grid).to(self.device)
        
        K = kernel(self.x, y, lens=self.lens, gain=self.gain, device=self.device)
        
        N = einsum('g k, g r -> r k', self.eig_vec, K)
        
        return N

"""Taken from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
Some modifications have been made to work with newer versions of Pytorch"""

import numpy as np
import torch
import torch.nn as nn


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    #Vc = torch.fft.rfft(v, 1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    
    k = - torch.arange(N, dtype=x.dtype,
                       device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def dct_shift(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    #Vc = torch.fft.rfft(v, 1)
    Vc = torch.view_as_real(torch.fft.fftshift(torch.fft.fft(v, dim=1), dim=1))
    
    k = - torch.arange(N, dtype=x.dtype,
                       device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype,
                     device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    #v = torch.fft.irfft(V, 1)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def idct_shift(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype,
                     device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    #v = torch.fft.irfft(V, 1)
    v = torch.fft.irfft(torch.fft.fftshift(torch.view_as_complex(V), dim=1), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


# Removed 2D/3D DCT functions as they are not needed for 1D data processing


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""

    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # don't learn this!


if __name__ == '__main__':
    x = torch.Tensor(1000, 4096)
    x.normal_(0, 1)
    linear_dct = LinearDCT(4096, 'dct')
    error = torch.abs(dct(x) - linear_dct(x))
    assert error.max() < 1e-3, (error, error.max())
    linear_idct = LinearDCT(4096, 'idct')
    error = torch.abs(idct(x) - linear_idct(x))
    assert error.max() < 1e-3, (error, error.max())

# Removed all image-related classes and functions (DCTBlur, Snow, rgb2hsv, hsv2rgb, rgb2lab, lab2rgb, etc.)
# as they are not needed for 1D data processing
from prettytable import PrettyTable

def count_parameters(model):
    """Count and display model parameters."""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
