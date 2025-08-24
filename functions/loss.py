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

def _trapz_weights(x: torch.Tensor) -> torch.Tensor:
    """Compute trapezoidal integration weights along last dim for 1D grids.

    Args:
        x: Tensor of shape (B, N) with sorted coordinates per sample.

    Returns:
        w: Tensor of shape (B, N) with trapezoidal weights.
    """
    # Ensure 2D (B, N)
    assert x.dim() == 2, "x must be (B, N)"
    dx = x[:, 1:] - x[:, :-1]  # (B, N-1)
    # Handle potential unsorted inputs gracefully by abs on dx
    dx = dx.abs()
    w = torch.zeros_like(x)
    # interior: 0.5*(dx_{i-1} + dx_i)
    if x.size(1) > 2:
        w[:, 1:-1] = 0.5 * (dx[:, 1:] + dx[:, :-1])
    # boundaries: half-intervals
    w[:, 0] = 0.5 * dx[:, 0]
    w[:, -1] = 0.5 * dx[:, -1]
    return w


def hilbert_loss_fn(model, sde, x_0, t, e, x_coord):
    """Coordinate-aware loss with MSE-like scaling.

    - Preserve coordinate awareness via trapezoidal weights for non-uniform grids.
    - Normalize weights to sum to 1 along the grid so the loss matches the
      numerical scale of the previous mean-squared-error (per-sample average).
    """
    # Forward SDE coefficients
    x_mean = sde.diffusion_coeff(t)
    noise = sde.marginal_std(t)

    # x_t construction and target epsilon (= -e)
    x_t = x_0 * x_mean[:, None] + e * noise.view(-1, 1)
    target = -e

    # Model expects two-channel input [signal, coord]
    model_input = torch.cat([x_t.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)
    output = model(model_input, t.float())

    # Coordinate-aware weighted average using trapezoidal weights
    # Normalize weights to sum to 1 so the loss scale matches MSE
    w = _trapz_weights(x_coord)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    loss = ((output - target).square() * w).sum(dim=1).mean(dim=0)
    return loss
