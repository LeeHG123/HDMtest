"""
Some codes are partially adapted from
https://github.com/AaltoML/generative-inverse-heat-dissipation/blob/main/scripts/datasets.py
"""
import numpy as np
import torch

from datasets.quadratic import QuadraticDataset
from datasets.gaussian  import GaussianDataset 
from datasets.doppler   import DopplerDataset  

def data_scaler(data):
    return data * 2. - 1.

def data_inverse_scaler(data):
    return (data + 1.) / 2.

def _get_adapt_params(config):
    subgroups = int(getattr(config.data, 'subgroups', 3))
    adapt = getattr(config.data, 'adapt', None)
    adapt = {} if adapt is None else adapt.__dict__ if hasattr(adapt, '__dict__') else dict(adapt)
    adapt_defaults = dict(
        pilot_points=2048, alpha=1.0, smooth_sigma=3.0,
        clip_quantile=0.98, min_per_group=1,
    )
    for k, v in adapt_defaults.items():
        adapt.setdefault(k, v)
    return subgroups, adapt

def get_dataset(config):
    grid_type = getattr(config.data, 'grid_type', 'uniform')
    subgroups, adapt = _get_adapt_params(config)

    if config.data.dataset == "Quadratic":
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                   num_points=config.data.dimension,
                                   seed=42,
                                   grid_type=grid_type,
                                   subgroups=subgroups,
                                   adapt_params=adapt)
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        subgroups=subgroups,
                                        adapt_params=adapt)
        test_dataset.is_train = False

    elif config.data.dataset == "Gaussian":
        dataset = GaussianDataset(num_data=config.data.num_data,
                                  num_points=config.data.dimension,
                                  seed=42,
                                  grid_type=grid_type,
                                  subgroups=subgroups,
                                  adapt_params=adapt)
        dataset.is_train = True
        test_dataset = GaussianDataset(num_data=config.data.num_data,
                                       num_points=config.data.dimension,
                                       seed=43,
                                       grid_type=grid_type,
                                       subgroups=subgroups,
                                       adapt_params=adapt)
        test_dataset.is_train = False

    elif config.data.dataset == "Doppler":
        dataset = DopplerDataset(num_data=config.data.num_data,
                                 num_points=config.data.dimension,
                                 seed=42,
                                 grid_type=grid_type,
                                 subgroups=subgroups,
                                 adapt_params=adapt)
        dataset.is_train = True
        test_dataset = DopplerDataset(num_data=config.data.num_data,
                                      num_points=config.data.dimension,
                                      seed=43,
                                      grid_type=grid_type,
                                      subgroups=subgroups,
                                      adapt_params=adapt)
        test_dataset.is_train = False
    else:
        raise NotImplementedError(f"Unknown dataset: {config.data.dataset}")
    return dataset, test_dataset



