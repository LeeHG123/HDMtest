import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import torchsde

def sampler(x, x_coord, *, model, sde, device, W, eps, dataset, steps=1000, sampler_input=None, method='srk'):
    """
    Sampler for 1D data using SDE-based generation
    
    Args:
        x: Initial noise tensor
        model: Score model
        sde: SDE object (VPSDE1D)
        device: Device to run on
        W: Noise generator (HilbertNoise)
        eps: Small epsilon value for numerical stability
        dataset: Dataset name (for gradient clipping)
        steps: Number of sampling steps
        sampler_input: Optional pre-generated noise
        method: Sampling method ('srk' for Stochastic Runge-Kutta, 'euler' for Euler-Maruyama)
    
    Returns:
        Generated samples
    """
    def sde_score_update(x, s, t):
        """
        input: x_s, s, t
        output: x_t
        """
        models = model(x, s)
        score_s = models * torch.pow(sde.marginal_std(s), -(2.0 - 1))[:, None].to(device)

        beta_step = sde.beta(s) * (s - t)
        x_coeff = 1 + beta_step / 2.0

        noise_coeff = torch.pow(beta_step, 1 / 2.0)
        if sampler_input == None:
            e = W.sample(x.shape)
        else:
            e = W.free_sample(free_input=sampler_input)

        score_coeff = beta_step
        x_t = x_coeff[:, None].to(device) * x + score_coeff[:, None].to(device) * score_s + noise_coeff[:, None].to(device) * e.to(device)

        return x_t

    if method == 'srk':
        return sampler_srk(x, x_coord, model=model, sde=sde, device=device, W=W, eps=eps, dataset=dataset, steps=steps, sampler_input=sampler_input)
    else:
        return sampler_euler(x, model, sde, device, W, eps, dataset, steps, sampler_input)


def sampler_euler(x, model, sde, device, W, eps, dataset, steps=1000, sampler_input=None):
    """
    Euler-Maruyama sampler for 1D data using SDE-based generation
    """
    def sde_score_update(x, s, t):
        """
        input: x_s, s, t
        output: x_t
        """
        models = model(x, s)
        score_s = models * torch.pow(sde.marginal_std(s), -(2.0 - 1))[:, None].to(device)

        beta_step = sde.beta(s) * (s - t)
        x_coeff = 1 + beta_step / 2.0

        noise_coeff = torch.pow(beta_step, 1 / 2.0)
        if sampler_input == None:
            e = W.sample(x.shape)
        else:
            e = W.free_sample(free_input=sampler_input)

        score_coeff = beta_step
        x_t = x_coeff[:, None].to(device) * x + score_coeff[:, None].to(device) * score_s + noise_coeff[:, None].to(device) * e.to(device)

        return x_t

    timesteps = torch.linspace(sde.T, eps, steps + 1).to(device)

    with torch.no_grad():
        for i in tqdm.tqdm(range(steps)):
            vec_s = torch.ones((x.shape[0],)).to(device) * timesteps[i]
            vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]

            x = sde_score_update(x, vec_s, vec_t)

            # Gradient clipping for stability
            size = x.shape
            l = x.shape[0]
            x = x.reshape((l, -1))
            indices = x.norm(dim=1) > 10
            if dataset == 'Gridwatch':
                x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 17
            else:
                x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 10
            x = x.reshape(size)

    return x


def sampler_srk(x, x_coord, *, model, sde, device, W, eps, dataset, steps=1000, sampler_input=None):
    """
    Stochastic Runge-Kutta sampler using torchsde
    """
    class ScoreBasedSDE(torch.nn.Module):
        def __init__(self, model, sde, x_coord):
            super().__init__()
            self.model = model
            self.sde = sde
            self.x_coord = x_coord.to(x.device)
            self.noise_type = "diagonal"
            self.sde_type = "ito"

        def f(self, t, y):
            # Drift term: using score function
            # Ensure t has correct shape for timestep embedding (1D tensor)
            if t.dim() == 0:
                t = t.unsqueeze(0).expand(y.shape[0])
            elif t.dim() > 1:
                t = t.squeeze()
            model_input = torch.cat([y.unsqueeze(1), self.x_coord.unsqueeze(1)], dim=1)
            models = self.model(model_input, t)
            score = models * torch.pow(self.sde.marginal_std(t), -(2.0 - 1))[:, None].to(y.device)
            beta_t = self.sde.beta(t)
            return -0.5 * beta_t[:, None].to(y.device) * (y + 2 * score)

        def g(self, t, y):
            # Diffusion term
            # Ensure t has correct shape for beta function
            if t.dim() == 0:
                t = t.unsqueeze(0).expand(y.shape[0])
            elif t.dim() > 1:
                t = t.squeeze()
            beta_t = self.sde.beta(t)
            return torch.sqrt(beta_t)[:, None].to(y.device) * torch.ones_like(y)

    # Create SDE instance
    sde_instance = ScoreBasedSDE(model, sde, x_coord)
    
    # Time points (forward time for torchsde)
    ts = torch.linspace(eps, sde.T, steps + 1).to(device)
    
    # Create reverse-time SDE for backward sampling
    class ReverseSDE(torch.nn.Module):
        def __init__(self, forward_sde):
            super().__init__()
            self.forward_sde = forward_sde
            self.noise_type = "diagonal"
            self.sde_type = "ito"

        def f(self, t, y):
            # Reverse time drift
            # Convert time for reverse sampling
            reverse_t = sde.T - t + eps
            if reverse_t.dim() == 0:
                reverse_t = reverse_t.unsqueeze(0).expand(y.shape[0])
            elif reverse_t.dim() > 1:
                reverse_t = reverse_t.squeeze()
            return -self.forward_sde.f(reverse_t, y)

        def g(self, t, y):
            # Same diffusion coefficient
            # Convert time for reverse sampling
            reverse_t = sde.T - t + eps
            if reverse_t.dim() == 0:
                reverse_t = reverse_t.unsqueeze(0).expand(y.shape[0])
            elif reverse_t.dim() > 1:
                reverse_t = reverse_t.squeeze()
            return self.forward_sde.g(reverse_t, y)
    
    reverse_sde = ReverseSDE(sde_instance)
    
    with torch.no_grad():
        # Use torchsde's SRK solver with forward time
        total_noise_increment = W.sample(x.shape).to(device) * (sde.T - eps)**0.5
        bm_interval = torchsde.BrownianInterval(
            t0=eps,
            t1=sde.T,
            size=x.shape,
            W=total_noise_increment,
            levy_area_approximation='space-time'
        )
        ys = torchsde.sdeint(reverse_sde, x, ts, method='srk', dt=1e-3, adaptive=False, bm=bm_interval)
        x = ys[-1]
        
        # Gradient clipping for stability
        size = x.shape
        l = x.shape[0]
        x = x.reshape((l, -1))
        indices = x.norm(dim=1) > 10
        if dataset == 'Gridwatch':
            x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 17
        else:
            x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 10
        x = x.reshape(size)

    return x