import torch
import torch.nn.functional as F

def loss_fn(model, sde, x_0, t, e):
    x_mean = sde.diffusion_coeff(x_0, t)
    noise = sde.marginal_std(e, t)

    x_t = x_mean + noise
    score = -noise

    output = model(x_t, t)

    loss = (output - score).square().sum(dim=(1,2,3)).mean(dim=0)
    return loss

def hilbert_loss_fn(model, sde, x_0, t, e):
    x_mean = sde.diffusion_coeff(t)
    noise = sde.marginal_std(t)

    x_t = x_0 * x_mean[:, None] + e * noise.view(-1, 1)
    score = -e

    output = model(x_t, t.float())

    loss = (output - score).square().sum(dim=(1)).mean(dim=0)
    return loss

def hilbert_sobolev_loss_fn(model, sde, x_0, t, e, sobolev_weight=1.0):
    """
    Computes the H¹ Sobolev norm-based loss for smoother 1D function generation.
    The loss is a weighted sum of the L² norm and the L² norm of the first derivative.
    The derivative is computed efficiently in the Fourier domain.

    Loss = ||output - score||_L²² + λ * ||∇(output - score)||_L²²
    
    Args:
        model: The score network.
        sde: The SDE object.
        x_0: The original data (ground truth functions).
        t: Timesteps.
        e: Noise sampled from Hilbert space.
        sobolev_weight (float): The weight (λ) for the derivative term.
    
    Returns:
        A scalar tensor representing the Sobolev loss.
    """
    # 1. Get model output (predicted score)
    x_mean = sde.diffusion_coeff(t)
    noise = sde.marginal_std(t)
    x_t = x_0 * x_mean[:, None] + e * noise.view(-1, 1)
    score = -e
    output = model(x_t, t.float())

    # 2. Calculate the difference between prediction and target
    diff = output - score  # Shape: (batch_size, num_points)

    # 3. Calculate the L² norm component of the loss
    l2_loss_term = diff.square().sum(dim=1)

    # 4. Calculate the H¹ norm component (derivative term) using Fourier differentiation
    num_points = diff.shape[1]
    
    # Apply Real Fast Fourier Transform for real-valued signals
    diff_fft = torch.fft.rfft(diff, dim=1)
    
    # Get the corresponding frequencies.
    freqs = torch.fft.rfftfreq(num_points, device=diff.device)
    freqs_sq = freqs.square()
    
    # Weight the squared magnitudes of FFT coefficients by squared frequencies
    h1_loss_term = (diff_fft.abs().square() * freqs_sq[None, :]).sum(dim=1)

    # 5. Combine the terms to get the final Sobolev loss
    sobolev_loss = (l2_loss_term + sobolev_weight * h1_loss_term).mean(dim=0)

    return sobolev_loss