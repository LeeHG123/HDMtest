### NUNO/NUFNO 1D Implementation Plan for HDM (Code Implementation Only)
================================================================================

#### Goal
--------------------------------------------------------------------------------
Replace `models/nufno.py` in the HDM codebase with a faithful 1D NUNO/NUFNO module (extendable to 2D/3D) that exactly follows the NUNO paper’s Algorithm 1 (K‑D tree domain decomposition with |D|·KL(P||U) selection and discrete equidistant split candidates) and the five‑stage pipeline:
non‑uniform points → (domain decomposition + forward interpolation) → uniform subdomain grids → P/Q linear projections + base FNO → backward interpolation → point‑space prediction.
**Constraint:** Keep `Hilbert_loss_fn` unchanged. The model must return point‑space predictions on the original coordinates so the existing HDM training loop and loss function continue to work without modification.
**Constraint:**  Preservation of models/nufno.py. The objective is to introduce a new, canonical implementation. The existing models/nufno.py file should not be modified or deleted, and can be kept for reference or comparison.

#### High‑Level Correctness Deltas vs. the Paper (what must be true)
--------------------------------------------------------------------------------
1) Domain decomposition must implement Algorithm 1 exactly:
   - At each iteration select D* = argmax_D |D|·KL(P||U; D).
   - Split along the axis with the largest bounding‑box scale (1D: the unique axis).
   - Choose the hyperplane x_k = b* that maximizes KL gain among Nmax equidistant candidates inside the current interval (exclude endpoints).
   - KL is computed by histogram density within the current bounding box; the reference Q is uniform on that box.
2) Interpolation/back‑interpolation are linear operators (ϕ linear):
   - Forward (point→grid) uses a linear, normalized triangular kernel (splat; “binning with linear weights”).
   - Backward (grid→point) uses the same kernel (gather), ensuring numerical consistency and preserving the decomposition “prediction error vs interpolation error” view.
3) Alignment of different subdomain grid sizes to a common length S_align is by resizing (default: FFT‑IFFT; option: linear up/down‑sampling). Handle Nyquist terms correctly.
4) Projections P, Q are location‑shared linear maps (1×1 conv / per‑position Linear), *no spatial mixing inside P/Q*. The base neural operator (FNO) performs the spatial mixing.
5) Loss is evaluated in point space after back‑interpolation. In HDM integration, we keep `Hilbert_loss_fn` intact by making the model’s `forward` return the predicted field on the original non‑uniform points.

#### Repository Layout (new/modified files)
--------------------------------------------------------------------------------
```
hdm/
  functions/
    domain_decomposition.py     # Algorithm 1 (K‑D tree) with serialization
    interpolation_1d.py         # Splat/Gather with precomputed maps
    resample_1d.py              # FFT and linear 1D resize
    nuno_cache.py               # Persistent cache for decomposition & maps
  models/
    nuno_framework.py           # NUFNO1D main module (P/Q/FNO wrapper)
  runner/
    hilbert_runner.py           # Model registry hook + precompute workflow
  datasets/
    quadratic.py, gaussian.py   # get_all_points(), fixed_geometry support
  configs/
    hdm_quadratic_nuno_fno.yml  # New hyperparameters
```

#### Configuration (example: configs/hdm_quadratic_nuno_fno.yml)
--------------------------------------------------------------------------------
```yaml
model:
  model_type: NUNO_FNO_1D
  n_subdomains: 8                   # number of leaves in K-D tree
  s_total: 4096                     # target sum of subdomain grid sizes
  s_min: 2                          # min grid points per subdomain
  n_bins_kl: 64                     # histogram bins for KL in 1D
  n_split_candidates: 5             # Nmax equidistant candidate split points
  align_size: 512                   # target S_align after resizing
  kernel: triangular                # linear kernel for splat/gather
  mix_subdomains: channel           # channel | independent
  fixed_geometry: true              # single decomposition for the dataset
  rebuild_every_k: 0                # if >0, rebuild decomposition every k steps (dynamic geometry)
  cache_decomposition: true
  cache_dir: ./.nuno_cache
  # FNO hyperparams (example; reuse existing FNO config when present)
  fno_width: 32
  fno_modes: 12
  fno_layers: 4

optim:
  amp: true
  grad_checkpoint: false
```

#### Algorithm 1: K‑D Tree Domain Decomposition (functions/domain_decomposition.py)
--------------------------------------------------------------------------------
**Data structures**
  `SubdomainDef`:
    - `idx`:      (N_s,)  indices of original points in this subdomain
    - `a, b`:     floats  interval bounds [a,b]
    - `s`:        int     assigned grid size s_s
    - `grid`:     (s,)    uniform nodes (optional; cacheable)

**Class `DomainDecomposer`**(`n_subdomains`, `n_bins_kl`, `n_split_candidates`, `s_total`, `s_min`, `seed`, ...)

**`fit(points_1d: (M,)) -> list[SubdomainDef]`**
  1) Initialize D0: `idx = arange(M)`, `[a0,b0] = [min(points), max(points)]`.
  2) Maintain a set `S` of active subdomains; start with `{D0}`.
  3) Repeat until `|S| = n_subdomains`:
       a) For each D in S, compute KL(P||U; D) from a histogram on `[a_D, b_D]`.
          - Use `N = min(n_bins_kl, max(1, |D|))` bins; uniform edges in `[a_D, b_D)`.
          - Add `ε` to empty bins (e.g., `ε=1e-12`) for numerical stability.
       b) Choose `D* = argmax_D |D| * KL(P||U; D)`.
       c) Split axis: in 1D it is x; in >1D pick the axis of max bounding‑box scale.
       d) Construct `Nmax = n_split_candidates` equidistant candidates in `(a*, b*)`
          (exclude endpoints), evaluate `Gain(D*, b)` for each:
            `Gain(D,b) = KL(D*) - (|D_>|/|D*|) KL(D_>) - (|D_≤|/|D*|) KL(D_≤)`
          where `D_> = {x in D*: x > b}`, `D_≤ = {x in D*: x ≤ b}`.
       e) Split D* at `b* = argmax_b Gain(D*, b)` into left/right; update `S`.
  4) Assign grid size to each final subdomain `s`:
       `s_s = max{s_min, round(s_total * |D_s| / M)}`.
  5) Precompute uniform nodes `grid_s = linspace(a_s, b_s, s_s)`.
  6) Sort subdomains by `a_s` (left→right) for stable keys & reproducibility.

**Auxiliary**:
  `transform_indices()` -> list of 1D index tensors for restoring original order.

**Complexity and determinism**
  Time: O(n_subdomains * M log M) (from candidate sorting), Space: O(M).
  Fix RNG seed and deterministic bin edges; ensure stable ties by leftmost rule.

#### 1D Interpolation (functions/interpolation_1d.py)
--------------------------------------------------------------------------------
**Kernel and grid geometry**
  `Δ = (b - a) / (s - 1)`, grid nodes `x_g = a + g Δ`, g=0,...,s-1.
  For a point `z_i`, define `g0 = floor((z_i - a)/Δ)`, `g1 = g0 + 1`.
  Weights: `w1 = (z_i - x_{g0})/Δ`, `w0 = 1 - w1`; mask if out of bounds.

**Precompute maps (per subdomain)**
  `SplatGatherMap1D`:
    - `g0, g1`:   (N_s,)  int32 indices
    - `w0, w1`:   (N_s,)  float32 weights
    - `mask0, mask1`: (N_s,) bool masks

**Forward interpolation (“splat”: points → grid)**
  Input:  `values (C, N_s)` → Output: `grid (C, s)`
  Compute numerator and denominator by `scatter_add` along `g0` and `g1`:
     `num[g0] += w0 * v`,   `num[g1] += w1 * v`
     `den[g0] += w0`,       `den[g1] += w1`
  Normalize: `u_g = num_g / (den_g + eps)`.  (eps ~ 1e-12)

**Backward interpolation (“gather”: grid → points)**
  Input: `grid (C, s)` → Output: `values_hat (C, N_s)`
  `û(z_i) = w0 * grid[:, g0] + w1 * grid[:, g1]`  (apply masks)

**Notes**
  - Use the same kernel for splat and gather (ϕ linear), ensuring consistency.
  - **Robust Handling of Empty Grid Cells**: The default normalization `u_g = num_g / (den_g + eps)` results in `u_g ≈ 0` for empty cells (`den_g ≈ 0`). This can discard information. A more robust (and configurable) approach should be implemented: if `den_g` is below a threshold, fall back to nearest-neighbor interpolation by copying the value from the closest non-empty grid cell. Log a warning if the ratio of empty cells exceeds a certain threshold (e.g., 20%) to indicate potential issues with the chosen `s_total` or `s_min`.

#### Resampling / Alignment to S_align (functions/resample_1d.py)
--------------------------------------------------------------------------------
**`fft_resample_1d(x, S_target): (B, C, S_src) → (B, C, S_tgt)`**
  - rfft → crop/zero‑pad spectrum → irfft.
  - Handle even/odd lengths and the Nyquist bin correctly (if even length, keep the Nyquist frequency real).
**`linear_resample_1d(x, S_target)`**: optional fallback for ablations.

#### NUFNO1D Module (models/nuno_framework.py)
--------------------------------------------------------------------------------
**Modes of subdomain coupling**
  a) **Channel‑stack (default)**: stack subdomain grids along channel axis
     → shape (B, C_in * n_sub, S_align). P/Q learn linear mixing across subdomains at each position.
  b) **Independent**: reshape to (B * n_sub, C_in, S_align), run FNO per subdomain, then merge. Useful for OOM scenarios.

**Forward signature**
  `forward(x_with_coords: (B, C_in+1, N), t: Optional[...] = None) -> (B, C_out, N)`
    1) Split: `coords = x_with_coords[:, -1]` (B,N), `values = x_with_coords[:, :-1]` (B,C_in,N).
    2) Fixed geometry (preferred for HDM benchmarks):
         - Build once: decomposition + SplatGatherMap per subdomain; persist via `nuno_cache`.
       Dynamic geometry:
         - If `rebuild_every_k>0`, recompute per k steps; LRU‑cache maps within a batch.
    3) For each subdomain `s`:
         - `idx_s`: gather `z_s`, `v_s` from `coords`/`values`.
         - Build or load `sg_map_s` and `(a_s, b_s, s_s)`.
         - `points_to_grid` → `grid_s (C_in, s_s)` → resample to `S_align`.
    4) Channel‑stack or Independent mode:
         - If channel‑stack: `concat` along channels → `(B, C_in*n_sub, S_align)`.
           `P`: 1×1 conv / Linear(shared)  maps `C_in*n_sub` → hidden (`n_h`).
           Run base FNO on `(B, n_h, S_align)`.
           `Q`: 1×1 conv / Linear(shared) maps `n_h` → `C_out*n_sub`.
         - If independent: apply FNO on `(B*n_sub, C_in, S_align)` and linear heads.
    5) Split subdomain outputs → (per‑sub) inverse‑resample to `s_s` → `grid_to_points` (gather) → `(B, C_out, N_s)` for each subdomain.
    6) Scatter back to original indices → `(B, C_out, N)`.  Return this tensor.
    7) (Optional) also return masks/aux stats for logging.

**Implementation details**
  - Keep index/weight tensors in FP32; enable autocast for signals to support AMP.
  - **Strict Role Separation of P, Q, and FNO**: It is critical to adhere to the NUNO design principle. The `P` (projection) and `Q` (query) layers must be strictly location-shared linear transformations (i.e., implemented as `1x1 Conv` or a `Linear` layer applied per-position). Their sole responsibility is **channel-wise mixing**. All **spatial mixing** (learning relationships across different positions) must be performed exclusively by the base neural operator (e.g., the FNO block via its spectral convolution). Using spatial kernels like `3x3 Conv` in P or Q is forbidden as it conflates the roles and violates the NUNO framework.
  - **Time Embedding Injection Point**: To maintain consistency with the existing HDM `FNO` architecture, the time embedding `t` should be processed and injected into the pipeline at the earliest meaningful stage. The correct sequence is:
    1.  Apply the `Lifting` layer to the input point-space values to map them to the hidden dimension `(B, C_in, N) -> (B, n_h, N)`.
    2.  Add the processed time embedding `temb` to these lifted features: `h = h_lifted + temb.unsqueeze(-1)`.
    3.  Perform the forward interpolation (`points_to_grid`) on these time-aware features `h`.
    This ensures that time-dependent information is available to all subsequent layers, including the base FNO.
  - Ensure permutation invariance w.r.t. input point ordering by relying solely on gather/scatter with saved original indices.

#### HDM Integration (keep Hilbert_loss_fn unchanged)
--------------------------------------------------------------------------------
**Contract**
  - **Do not change** `Hilbert_loss_fn` or the trainer. Ensure that:
    * The model registry adds "NUNO_FNO_1D" and returns a module whose forward signature and output shape match existing consumers of `models/nufno.py`.
    * The `forward` method returns predictions at the original non‑uniform coordinates `(B, C_out, N)`, so any downstream loss (incl. `Hilbert_loss_fn`) continues to work verbatim.
  - If the old `nufno.py` returned grid‑space tensors, add a tiny adapter layer *inside the model* to back‑interpolate to point‑space before returning.
  - Preserve dtype/device behavior already expected by the runner (AMP safe).

#### Caching and Runner
--------------------------------------------------------------------------------
**`runner/hilbert_runner.py`**
  - Register `model_type == "NUNO_FNO_1D"`.
  - `fixed_geometry=true`:
      `dataset.get_all_points() → model.setup_decomposition()`
      Persist decomposition + sg‑maps + grids via `nuno_cache` under a hash of (dataset_id, config, seed).
  - `dynamic geometry`:
      `rebuild_every_k>0`: recompute periodically; cache per batch with an LRU (cap by memory).

**`datasets/*`**
  - Add `get_all_points()` for fixed‑geometry precomputation.
  - Optional: precompute and return per‑sample metadata (subdomain indices, [a,b], s, sg‑map keys) to amortize startup.

#### Logging, Metrics, and Losses
--------------------------------------------------------------------------------
**Loss (unchanged usage)**:
  - Compute loss in point space using existing `Hilbert_loss_fn` exactly as before.

#### Performance & Stability
--------------------------------------------------------------------------------
**Memory**
  - Chunk splat/gather over subdomains and/or channels to avoid OOM.
  - For channel‑stack, allow grad‑checkpoint around P/FNO/Q to trade compute for memory.

**Speed**
  - For fixed geometry, precompute everything feasible (subdomain indices, maps) and perform pure gather/scatter in the training loop.
  - Batch all per‑subdomain FFT resizes by stacking subdomains along channels to minimize calls.

**Determinism & Reproducibility**
  - Deterministic histogram bin edges, candidate split generation, and tie‑breakers.
  - Expose flags to set `torch/cudnn` deterministic modes.

**AMP/DDP**
  - Keep maps in FP32; signals under autocast.
  - In DDP, only rank0 writes cache; barrier then loads on all ranks.

#### Failure Modes & Guardrails
--------------------------------------------------------------------------------
- **Excess empty cells**:
    Increase `s_min` or reduce `s_total`; detect empty‑ratio threshold and auto‑adjust.
- **Unstable splitting**:
    Increase `n_split_candidates` and `n_bins_kl`; apply `ε`‑smoothing in KL.
- **Boundary leakage**:
    Assertions when subdomain boundary masks are violated; log boundary jumps.
- **OOM**:
    Fall back to Independent mode; reduce `align_size` or FNO width/modes.

#### Pseudocode Snippets (reference)
--------------------------------------------------------------------------------
**`DomainDecomposer.fit`**
```python
S=[D0]; while len(S) < n_sub:
  j = argmax_D |D|·KL(P||U; D)
  B = linspace(a_j, b_j, n_cand+2)[1:-1]      # equidistant candidates
  b_star = argmax_b Gain(D_j, b)
  S.remove(D_j); S.add(left(D_j,b_star)); S.add(right(D_j,b_star))
for s in S:
  s.s = max(s_min, round(s_total * |D_s| / M))
  s.grid = linspace(a_s, b_s, s.s)
```

**`points_to_grid(values, sg_map, s)`**:
  scatter_add for numerator/denominator; then normalize.

**`grid_to_points(grid, sg_map)`**:
  `return w0*grid[..., g0] + w1*grid[..., g1]`

**`fft_resample_1d(x, S_target)`**:
  rfft → crop/pad → irfft  (careful with even-length Nyquist bin)

#### Paper‑faithfulness Checklist (to verify during review)
--------------------------------------------------------------------------------
[ ] |D|·KL selection & longest‑axis splitting
[ ] Discrete equidistant split candidates (Nmax) and Gain maximization
[ ] Histogram KL on the bounding box; N_bins ≤ |D| with ε‑smoothing
[ ] Linear ϕ (same kernel for splat/gather); point‑space loss
[ ] Alignment via resize (FFT default) to a common `S_align`
[ ] Location‑shared P/Q (1×1 conv/Linear), spatial mixing in base FNO only
[ ] Outputs back‑interpolated to original points so `Hilbert_loss_fn` remains unchanged