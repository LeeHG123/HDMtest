import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class SubdomainDef:
    idx: torch.Tensor        # (N_s,) long tensor of original indices
    a: float                 # left bound
    b: float                 # right bound
    s: int                   # assigned grid size
    grid: Optional[torch.Tensor] = None  # (s,) uniform nodes


def _hist_kl(points: torch.Tensor, a: float, b: float, n_bins: int, eps: float = 1e-12) -> float:
    """
    Compute KL(P||U) for points within [a,b] using uniform bins.
    points: (M_s,) 1D tensor on CPU or CUDA
    Returns float (python float)
    """
    if points.numel() == 0 or not math.isfinite(a) or not math.isfinite(b) or b <= a:
        return 0.0

    M = points.numel()
    n_bins = max(1, min(int(n_bins), int(M)))
    # Create bin edges uniformly in [a,b)
    edges = torch.linspace(a, b, steps=n_bins + 1, device=points.device)

    # Digitize points to bins [0..n_bins-1], last edge exclusive except the rightmost equal to b maps to n_bins-1
    # torch.bucketize returns indices in [0..n_bins], we map to [0..n_bins-1]
    bin_ix = torch.bucketize(points, edges, right=False)
    bin_ix = torch.clamp(bin_ix - 1, 0, n_bins - 1)

    # Histogram counts
    counts = torch.bincount(bin_ix, minlength=n_bins).to(points.dtype)
    p = counts / (M + 0.0)

    # Add epsilon to avoid log(0)
    p_safe = torch.clamp(p, min=eps)
    q = 1.0 / n_bins
    # KL(P||U) = sum p * log(p / q)
    kl = (p_safe * torch.log(p_safe / q)).sum()
    return float(kl.detach().cpu())


def _gain_for_split(points: torch.Tensor, a: float, b: float, n_bins: int, b_split: float, eps: float = 1e-12) -> float:
    """
    Gain(D, b) = KL(D) - (|D_>|/|D|) KL(D_>) - (|D_<=|/|D|) KL(D_<=)
    points: (M_s,)
    """
    M = points.numel()
    if M == 0:
        return 0.0
    kl_full = _hist_kl(points, a, b, n_bins, eps)
    left_mask = points <= b_split
    right_mask = ~left_mask
    L = int(left_mask.sum())
    R = M - L
    if L == 0 or R == 0:
        # Degenerate split has zero gain
        return -1e9
    kl_left = _hist_kl(points[left_mask], a, b_split, n_bins, eps)
    kl_right = _hist_kl(points[right_mask], b_split, b, n_bins, eps)
    gain = kl_full - (R / M) * kl_right - (L / M) * kl_left
    return gain


class DomainDecomposer:
    """
    Paper-faithful 1D domain decomposition (Algorithm 1).
    Builds a K-D tree (1D: interval splits) by maximizing |D| * KL(P||U; D)
    and choosing the best split among equidistant candidates.
    """

    def __init__(
        self,
        n_subdomains: int,
        n_bins_kl: int = 64,
        n_split_candidates: int = 5,
        s_total: int = 4096,
        s_min: int = 2,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.n_subdomains = int(n_subdomains)
        self.n_bins_kl = int(n_bins_kl)
        self.n_split_candidates = int(n_split_candidates)
        self.s_total = int(s_total)
        self.s_min = int(s_min)
        self.seed = seed
        self.rng = torch.Generator(device='cpu')
        if seed is not None:
            self.rng.manual_seed(int(seed))
        self.device = device

    def fit(self, points_1d: torch.Tensor) -> List[SubdomainDef]:
        """
        points_1d: (M,) 1D tensor, arbitrary device. Returns list of SubdomainDef.
        """
        if points_1d.dim() != 1:
            raise ValueError("points_1d must be a 1D tensor")
        device = points_1d.device
        M = points_1d.numel()
        if M == 0:
            return []

        # Initial domain D0
        idx0 = torch.arange(M, device=device, dtype=torch.long)
        a0 = float(points_1d.min())
        b0 = float(points_1d.max())

        S: List[Tuple[torch.Tensor, float, float]] = [(idx0, a0, b0)]

        # Helper to compute priority: |D| * KL(D)
        def importance(idx: torch.Tensor, a: float, b: float) -> float:
            pts = points_1d[idx]
            kl = _hist_kl(pts, a, b, n_bins=self.n_bins_kl)
            return float(pts.numel()) * kl

        # Build until reaching desired number of leaves
        while len(S) < self.n_subdomains:
            # Select D* = argmax |D| * KL(D)
            scores = [importance(ix, a, b) for (ix, a, b) in S]
            j = int(torch.tensor(scores).argmax().item())
            idx_star, a_star, b_star = S[j]

            # Degenerate interval or singleton => stop splitting this domain
            if idx_star.numel() <= 1 or not math.isfinite(a_star) or not math.isfinite(b_star) or (b_star - a_star) <= 0:
                # Cannot split further; duplicate to keep counts growing
                break

            # Split candidates: equidistant in (a*, b*) excluding endpoints
            n_cand = max(1, self.n_split_candidates)
            cand = torch.linspace(a_star, b_star, steps=n_cand + 2, device=device)[1:-1]

            # Evaluate gain for each candidate; pick best
            pts_star = points_1d[idx_star]
            gains = [
                _gain_for_split(pts_star, a_star, b_star, self.n_bins_kl, float(bc))
                for bc in cand
            ]
            best_i = int(torch.tensor(gains).argmax().item())
            b_split = float(cand[best_i])

            # Partition indices
            left_mask = pts_star <= b_split
            right_mask = ~left_mask
            idx_left = idx_star[left_mask]
            idx_right = idx_star[right_mask]

            # Stable: left first, then right (sorted by a)
            S.pop(j)
            S.append((idx_left, a_star, b_split))
            S.append((idx_right, b_split, b_star))

            if len(S) >= self.n_subdomains:
                break

        # Assign grid sizes proportional to |D_s|
        total = sum(int(ix.numel()) for (ix, _, _) in S)
        subdomains: List[SubdomainDef] = []
        for (ix, a, b) in S:
            frac = (ix.numel() / max(1, total))
            s_s = max(self.s_min, int(round(self.s_total * frac)))
            grid = torch.linspace(float(a), float(b), steps=s_s, device=device)
            subdomains.append(SubdomainDef(idx=ix.long(), a=float(a), b=float(b), s=int(s_s), grid=grid))

        # Sort by a for reproducibility
        subdomains.sort(key=lambda sd: sd.a)
        return subdomains

    @staticmethod
    def transform_indices(subdomains: List[SubdomainDef]) -> List[torch.Tensor]:
        return [sd.idx for sd in subdomains]

