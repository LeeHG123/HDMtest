import numpy as np
import torch
import math


class VPSDE1D:

    def __init__(self, schedule='cosine'):
        self.beta_0 = 0.01
        self.beta_1 = 20
        self.cosine_s = 0.008
        self.schedule = schedule
        self.cosine_beta_max = 999.

        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. \
                            * (1. + self.cosine_s) / math.pi - self.cosine_s

        if schedule == 'cosine':
            self.T = 0.9946
        else:
            self.T = 1.

        self.sigma_min = 0.01
        self.sigma_max = 20
        self.eps = 1e-5
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))

    def beta(self, t):
        if self.schedule == 'linear':
            beta = (self.beta_1 - self.beta_0) * t + self.beta_0
        elif self.schedule == 'cosine':
            beta = math.pi / 2 * 2 / (self.cosine_s + 1) * torch.tan(
                (t + self.cosine_s) / (1 + self.cosine_s) * math.pi / 2)
        else:
            beta = 2 * np.log(self.sigma_max / self.sigma_min) * (t * 0 + 1)

        return beta

    def marginal_log_mean_coeff(self, t):
        if self.schedule == 'linear':
            log_alpha_t = - 1 / (2 * 2) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / 2 * t * self.beta_0

        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(
                torch.clamp(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.), -1, 1))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0

        else:
            log_alpha_t = -torch.exp(np.log(self.sigma_min) + t * np.log(self.sigma_max / self.sigma_min))

        return log_alpha_t

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.pow(1. - torch.exp(self.marginal_log_mean_coeff(t) * 2), 1 / 2)

    def inverse_a(self, a):
        return 2 / np.pi * (1 + self.cosine_s) * torch.acos(a) - self.cosine_s