"""
Some codes are partially adapted from
https://github.com/AaltoML/generative-inverse-heat-dissipation/blob/main/scripts/datasets.py
"""
import numpy as np
import torch

from datasets.quadratic import QuadraticDataset
from datasets.gaussian  import GaussianDataset  

def data_scaler(data):
    return data * 2. - 1.

def data_inverse_scaler(data):
    return (data + 1.) / 2.

def get_dataset(config):
    grid_type = getattr(config.data, 'grid_type', 'uniform')
    noise_std = float(getattr(config.data, 'noise_std', 0.0))
    name = str(config.data.dataset)

    if name == "Quadratic":
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                   num_points=config.data.dimension,
                                   seed=42,
                                   grid_type=grid_type,
                                   noise_std=noise_std,
                                   func_type='quadratic')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='quadratic')
        test_dataset.is_train = False

    elif name == "Linear" or name.lower() == "linear":   
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                   num_points=config.data.dimension,
                                   seed=42,
                                   grid_type=grid_type,
                                   noise_std=noise_std,
                                   func_type='linear')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='linear')
        test_dataset.is_train = False

    elif name == "Circle":
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                   num_points=config.data.dimension,
                                   seed=42,
                                   grid_type=grid_type,
                                   noise_std=noise_std,
                                   func_type='circle')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='circle')
        test_dataset.is_train = False      

    elif name == "Sin" or name.lower() == "sin":   
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                   num_points=config.data.dimension,
                                   seed=42,
                                   grid_type=grid_type,
                                   noise_std=noise_std,
                                   func_type='sin')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='sin')
        test_dataset.is_train = False          

    elif name == "Sinc" or name.lower() == "sinc":
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                num_points=config.data.dimension,
                                seed=42,
                                grid_type=grid_type,
                                noise_std=noise_std,
                                func_type='sinc')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='sinc')
        test_dataset.is_train = False

    elif name.lower() in ("matern", "matern32", "matern_32"):
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                num_points=config.data.dimension,
                                seed=42,
                                grid_type=grid_type,
                                noise_std=noise_std, 
                                func_type='matern32')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='matern32')
        test_dataset.is_train = False

    elif name.lower() in ("matern12", "matern_12"):
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                num_points=config.data.dimension,
                                seed=42,
                                grid_type=grid_type,
                                noise_std=noise_std, 
                                func_type='matern12')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='matern12')
        test_dataset.is_train = False

    elif name.lower() in ("matern52", "matern_52"):
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                num_points=config.data.dimension,
                                seed=42,
                                grid_type=grid_type,
                                noise_std=noise_std, 
                                func_type='matern52')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='matern52')
        test_dataset.is_train = False

    elif name.lower() in ("rq", "rationalquadratic", "rational_quadratic"):
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                num_points=config.data.dimension,
                                seed=42,
                                grid_type=grid_type,
                                noise_std=noise_std, 
                                func_type='rq')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='rq')
        test_dataset.is_train = False        

    elif name.lower() in ("blocks", "block"):                   
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                   num_points=config.data.dimension,
                                   seed=42,
                                   grid_type=grid_type,
                                   noise_std=noise_std,
                                   func_type='blocks')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='blocks')
        test_dataset.is_train = False        

    elif name == "Gaussian":
        dataset = GaussianDataset(num_data=config.data.num_data,
                                  num_points=config.data.dimension,
                                  seed=42,
                                  grid_type=grid_type,
                                  noise_std=noise_std)
        dataset.is_train = True
        test_dataset = GaussianDataset(num_data=config.data.num_data,
                                       num_points=config.data.dimension,
                                       seed=43,
                                       grid_type=grid_type,
                                       noise_std=noise_std)
        test_dataset.is_train = False

    elif name == "Doppler":
        dataset = QuadraticDataset(num_data=config.data.num_data,
                                   num_points=config.data.dimension,
                                   seed=42,
                                   grid_type=grid_type,
                                   noise_std=noise_std,
                                   func_type='doppler')
        dataset.is_train = True
        test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                        num_points=config.data.dimension,
                                        seed=43,
                                        grid_type=grid_type,
                                        noise_std=noise_std,
                                        func_type='doppler')
        test_dataset.is_train = False
    else:
        raise NotImplementedError(f"Unknown dataset: {config.data.dataset}")
    return dataset, test_dataset



