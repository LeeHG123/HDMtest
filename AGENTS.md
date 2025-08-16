# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: CLI entry. Parses config from `configs/` and starts training/sampling via `runner.HilbertDiffusion`.
- `configs/`: YAML configs (e.g., `hdm_quadratic_fno.yml`).
- `runner/`: Training/evaluation loop and checkpointing.
- `datasets/`: Synthetic datasets (e.g., `QuadraticDataset`, `GaussianDataset`).
- `models/`: Backbones (FNO/AFNO/MLP variants).
- `functions/`: Samplers, SDEs, losses, utilities.
- `evaluate/`: MMD power test utilities.
- Outputs: created under `--exp` (e.g., `outs/`), with `logs/`, `samples/`, and `tensorboard/` subfolders.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt` (Python 3.8 recommended).
- Single‑GPU train: `python main.py --config hdm_quadratic_fno.yml --exp outs/quadratic_experiment`.
- Multi‑GPU (DDP): `torchrun --nproc_per_node=2 main.py --config hdm_quadratic_fno.yml --exp outs/quadratic_experiment --distributed`.
- Sample from a checkpoint: `python main.py --config hdm_quadratic_fno.yml --exp outs/quadratic_experiment --sample --ckpt_step 20000 --sample_type ode --nfe 1000`.
- TensorBoard: `tensorboard --logdir outs/tensorboard`.

## Coding Style & Naming Conventions
- Python style: PEP 8, 4‑space indent, prefer type hints on public APIs.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.
- Configs: `hdm_<dataset>_<arch>.yml` (e.g., `hdm_quadratic_fno.yml`).
- Logging: use `logging` (already configured in `main.py`); avoid `print` in library code.

## Testing Guidelines
- No formal unit tests present. Use quick smoke runs:
  - Train: lower config sizes if needed, then run with `--seed 42` and `--verbose debug`.
  - Sampling: reduce cost via `--nfe 10` and a smaller batch in config.
- Evaluation: use `evaluate/power.py` utilities from a notebook/script to compute MMD‑based power.

## Commit & Pull Request Guidelines
- Commits: imperative mood with scope, e.g., `runner: fix gradient clipping`; explain why, not just what.
- PRs: include description, linked issue, repro commands, and (when relevant) TensorBoard screenshots; ensure configs and paths are valid and don’t commit large artifacts.

## Security & Configuration Tips
- Don’t commit datasets/checkpoints; keep `outs/`, `exp/`, and `*.pth` out of VCS.
- Prefer `--exp` for outputs; avoid hard‑coded absolute paths.
- Set seeds (`--seed`) for reproducibility; `--distributed` uses `torch.distributed` env vars automatically.
