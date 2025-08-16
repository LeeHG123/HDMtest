import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from functions.domain_decomposition import DomainDecomposer, SubdomainDef
from functions.interpolation_1d import SplatGatherMap1D, build_sg_map, points_to_grid, grid_to_points
from functions.resample_1d import fft_resample_1d, linear_resample_1d
from functions.nuno_cache import NunoCache, make_key
from .fno import FNO


class NUFNO1D(nn.Module):
    """
    NUNO/NUFNO 1D.

    - Performs domain decomposition (K-D tree in 1D), splat to per-subdomain uniform grids,
      resamples to a common aligned size, applies P (1x1 conv) -> FNO -> Q (1x1 conv),
      then back-interpolates to the original point coordinates.

    Forward(x_with_coords, t): x_with_coords (B, C_in, N) where last channel is coords (normalized),
                               returns (B, N) prediction at original points.
    """

    def __init__(self, config) -> None:
        super().__init__()
        cfg = config.model

        # Decomposition / mapping cfg
        self.n_subdomains = int(getattr(cfg, 'n_subdomains', 8))
        self.s_total = int(getattr(cfg, 's_total', 4096))
        self.s_min = int(getattr(cfg, 's_min', 2))
        self.n_bins_kl = int(getattr(cfg, 'n_bins_kl', 64))
        self.n_split_candidates = int(getattr(cfg, 'n_split_candidates', 5))
        self.align_size = int(getattr(cfg, 'align_size', 512))
        self.kernel = str(getattr(cfg, 'kernel', 'triangular'))
        self.mix_subdomains = str(getattr(cfg, 'mix_subdomains', 'channel'))  # 'channel' | 'independent'
        self.fixed_geometry = bool(getattr(cfg, 'fixed_geometry', True))
        self.rebuild_every_k = int(getattr(cfg, 'rebuild_every_k', 0))
        self.cache_decomposition = bool(getattr(cfg, 'cache_decomposition', True))
        self.cache_dir = str(getattr(cfg, 'cache_dir', './.nuno_cache'))
        self.coord_scale = float(getattr(cfg, 'coord_scale', 10.0))  # undo normalization from runner

        # Channel config
        self.c_in_signal = int(getattr(cfg, 'c_in_signal', 1))  # number of signal channels (exclude coord channel)
        self.c_out = int(getattr(cfg, 'c_out', 1))

        # Base FNO hyperparams
        fno_width = int(getattr(cfg, 'fno_width', 32))
        fno_modes = int(getattr(cfg, 'fno_modes', 12))
        fno_layers = int(getattr(cfg, 'fno_layers', 4))

        # P/Q are 1x1 convs (per-position linear maps)
        if self.mix_subdomains == 'channel':
            in_p = self.c_in_signal * self.n_subdomains
            out_q = self.c_out * self.n_subdomains
        else:  # independent
            in_p = self.c_in_signal
            out_q = self.c_out

        self.P = nn.Conv1d(in_channels=in_p, out_channels=fno_width, kernel_size=1)
        self.Q = nn.Conv1d(in_channels=fno_width, out_channels=out_q, kernel_size=1)

        # Base FNO performs spatial mixing on uniform aligned grids
        self.base = FNO(
            n_modes=(fno_modes,),
            hidden_channels=fno_width,
            in_channels=fno_width,
            out_channels=fno_width,
            lifting_channels=fno_width,
            projection_channels=fno_width,
            n_layers=fno_layers,
            separable=True,
            preactivation=True,
            use_mlp=True,
        )

        # Decomposition and maps (populated by setup_decomposition or lazily)
        self.decomposer = DomainDecomposer(
            n_subdomains=self.n_subdomains,
            n_bins_kl=self.n_bins_kl,
            n_split_candidates=self.n_split_candidates,
            s_total=self.s_total,
            s_min=self.s_min,
            seed=getattr(config, 'seed', None),
        )
        self.subdomains: Optional[List[SubdomainDef]] = None
        self.sg_maps: Optional[List[SplatGatherMap1D]] = None
        self.cache = NunoCache(self.cache_dir)
        self._steps = 0

        # Resampling method
        self._resample = fft_resample_1d

    def setup_decomposition(self, points_1d: torch.Tensor) -> None:
        """
        Precompute domain decomposition and splat/gather maps for fixed geometry.
        points_1d: (N,) real coordinates (not normalized)
        """
        points_1d = points_1d.detach().to('cpu').view(-1)
        key_cfg = dict(
            n_subdomains=self.n_subdomains,
            s_total=self.s_total,
            s_min=self.s_min,
            n_bins_kl=self.n_bins_kl,
            n_split_candidates=self.n_split_candidates,
        )
        key = make_key(points_1d, key_cfg)
        loaded = self.cache.load(key) if self.cache_decomposition else None
        if loaded is not None:
            logging.info("Loaded NUNO decomposition from cache")
            self._load_from_cache_bundle(loaded)
            return

        # Fit decomposer and build maps
        subdomains = self.decomposer.fit(points_1d)
        sg_maps = [build_sg_map(points_1d[sd.idx], sd.a, sd.b, sd.s) for sd in subdomains]
        self.subdomains = subdomains
        self.sg_maps = sg_maps

        if self.cache_decomposition:
            bundle = self._make_cache_bundle()
            self.cache.save(key, bundle)

    def _make_cache_bundle(self) -> Dict[str, any]:
        assert self.subdomains is not None and self.sg_maps is not None
        # Store tensors necessary to rebuild maps
        bundle = dict(
            subdomains=[
                dict(idx=sd.idx.cpu(), a=sd.a, b=sd.b, s=sd.s, grid=sd.grid.cpu() if sd.grid is not None else None)
                for sd in self.subdomains
            ]
        )
        return bundle

    def _load_from_cache_bundle(self, bundle: Dict[str, any]) -> None:
        subs = []
        maps = []
        # We cannot rebuild sg_maps without original points; the cached bundle here only restores subdomain defs.
        for ent in bundle['subdomains']:
            sd = SubdomainDef(
                idx=ent['idx'].long(),
                a=float(ent['a']),
                b=float(ent['b']),
                s=int(ent['s']),
                grid=ent['grid'] if ent['grid'] is None else ent['grid'].float(),
            )
            subs.append(sd)
        self.subdomains = subs
        self.sg_maps = None  # rebuilt on first forward with batch coords

    def _ensure_maps(self, coords_real_1st: torch.Tensor) -> None:
        """
        Ensure sg_maps exist; if absent (e.g., loaded from cache without maps), rebuild
        using coords from the first sample.
        """
        if self.subdomains is None:
            # Build fresh decomposition from this batch (use first sample geometry)
            self.subdomains = self.decomposer.fit(coords_real_1st)
        if self.sg_maps is None:
            self.sg_maps = [build_sg_map(coords_real_1st[sd.idx], sd.a, sd.b, sd.s) for sd in self.subdomains]

    def _maybe_rebuild_dynamic(self, coords_real_1st: torch.Tensor) -> None:
        if self.rebuild_every_k > 0 and (self._steps % self.rebuild_every_k == 0):
            self.subdomains = self.decomposer.fit(coords_real_1st)
            self.sg_maps = [build_sg_map(coords_real_1st[sd.idx], sd.a, sd.b, sd.s) for sd in self.subdomains]

    def forward(self, x_with_coords: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_with_coords: (B, C_in, N), last channel is normalized coords x/10.0
        t: (B,) or None
        Returns: (B, N)
        """
        B, C_in, N = x_with_coords.shape
        device = x_with_coords.device
        # split channels
        coords_norm = x_with_coords[:, -1, :]  # (B,N)
        signals = x_with_coords[:, : C_in - 1, :]  # (B, C_sig, N)

        # Undo normalization for decomposition space
        coords_real = coords_norm * self.coord_scale  # (B,N)
        coords_first = coords_real[0].detach()

        # Precompute or ensure maps
        if self.fixed_geometry:
            self._ensure_maps(coords_first)
        else:
            self._maybe_rebuild_dynamic(coords_first)
            self._ensure_maps(coords_first)

        assert self.subdomains is not None and self.sg_maps is not None

        # Per-subdomain splat → aligned length
        per_sub_grids: List[torch.Tensor] = []  # each (B, C_sig, S_align)
        sizes: List[int] = []
        for sd, sg in zip(self.subdomains, self.sg_maps):
            # Gather values for this subdomain
            v_s = signals.index_select(dim=2, index=sd.idx)
            # points->grid
            grid_s = points_to_grid(v_s, sg, sd.s)  # (B, C_sig, s_s)
            # resize to align
            grid_s_aligned = self._resample(grid_s, self.align_size)
            per_sub_grids.append(grid_s_aligned)
            sizes.append(sd.s)

        if self.mix_subdomains == 'channel':
            x_cat = torch.cat(per_sub_grids, dim=1)  # (B, C_sig*n_sub, S_align)
            h = self.P(x_cat)  # (B, fno_width, S_align)
            h = self.base(h, t if t is not None else torch.zeros(B, device=device))  # (B, fno_width, S_align)
            y = self.Q(h)  # (B, C_out*n_sub, S_align)
            # Split back per subdomain
            chunks = torch.split(y, split_size_or_sections=self.c_out, dim=1)
        else:  # independent
            x_stack = torch.stack(per_sub_grids, dim=1)  # (B, n_sub, C_sig, S_align)
            Bs, n_sub, Csig, S = x_stack.shape
            x_flat = x_stack.view(Bs * n_sub, Csig, S)
            h = self.P(x_flat)  # (B*n_sub, fno_width, S)
            h = self.base(h, t.repeat_interleave(n_sub) if t is not None else torch.zeros(Bs * n_sub, device=device))
            y_flat = self.Q(h)  # (B*n_sub, C_out, S)
            y = y_flat.view(Bs, n_sub, self.c_out, S)
            chunks = [y[:, i] for i in range(n_sub)]  # list of (B, C_out, S)

        # Back to points per subdomain and scatter to full output
        out = torch.zeros(B, self.c_out, N, device=device, dtype=signals.dtype)
        for sd, sg, y_align in zip(self.subdomains, self.sg_maps, chunks):
            # Resize aligned grid back to subdomain size, then gather to points
            if y_align.dim() == 3:  # (B,C_out,S_align)
                y_resized = linear_resample_1d(y_align, sd.s)  # (B,C_out,s)
            else:  # safety
                y_resized = y_align
            vals_hat = grid_to_points(y_resized, sg)  # (B,C_out,N_s)
            # Scatter to output positions
            out[:, :, sd.idx] = vals_hat

        self._steps += 1
        return out.squeeze(1)
