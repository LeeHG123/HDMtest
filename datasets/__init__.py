"""
Some codes are partially adapted from
https://github.com/AaltoML/generative-inverse-heat-dissipation/blob/main/scripts/datasets.py
"""
import numpy as np
import torch

from datasets.quadratic import QuadraticDataset

def data_scaler(data):
    return data * 2. - 1.

def data_inverse_scaler(data):
    return (data + 1.) / 2.

_QUADRATIC_FAMILY = {
    "Quadratic": "quadratic",
    "Linear": "linear",
    "Circle": "circle",
    "Sin": "sin",
    "Sinc": "sinc",
    "Doppler": "doppler",
    "GaussianBumps": "gaussian_bumps",
    "AMSin": "am_sin",
}

def get_dataset(config):
    name = str(config.data.dataset)

    if name in _QUADRATIC_FAMILY:
        func_type = _QUADRATIC_FAMILY[name]

        dataset = QuadraticDataset(
            num_data=config.data.num_data,
            num_points=config.data.dimension,
            seed=getattr(config.data, "seed", 42),
            grid_type=config.data.grid_type,
            noise_std=config.data.noise_std,
            func_type=func_type,
        )
        test_dataset = QuadraticDataset(
            num_data=config.data.num_data,
            num_points=config.data.dimension,
            seed=getattr(config.data, "seed", 43),
            grid_type=config.data.grid_type,
            noise_std=config.data.noise_std,
            func_type=func_type,
        )

        dataset.is_train = True
        test_dataset.is_train = False
        return dataset, test_dataset

    raise NotImplementedError(
        f"Unknown dataset '{name}'. Supported: {list(_QUADRATIC_FAMILY.keys())}"
    )



