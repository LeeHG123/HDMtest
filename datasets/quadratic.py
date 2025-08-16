import torch

class QuadraticDataset(torch.utils.data.Dataset):
    def __init__(self, num_data, num_points, seed=42, grid_type='uniform'):
        super().__init__()
        torch.manual_seed(seed)

        self.num_data = num_data
        self.num_points = num_points
        self.seed = seed
        self.grid_type = grid_type
        # Default: training dataset unless overwritten in get_dataset
        self.is_train = True
        
        if grid_type == 'uniform':
            x_base = torch.linspace(start=-10., end=10., steps=self.num_points)
        elif grid_type == 'random':
            # Base grid used to initialize Hilbert noise and for validation if needed.
            # For training with random grids, per-item grids will be generated in __getitem__.
            x_base = (torch.rand(self.num_points) * 20 - 10).sort().values
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'. Choose 'uniform' or 'random'.")
            
        self.x = x_base.unsqueeze(0).repeat(self.num_data, 1)
        self.dataset = self._create_dataset()

    def _create_dataset(self):
        torch.manual_seed(self.seed)
        a = torch.randint(low=0, high=2, size=(self.x.shape[0], 1)).repeat(1, self.num_points) * 2 - 1
        eps = torch.normal(mean=0., std=1., size=(self.x.shape[0], 1)).repeat(1, self.num_points)

        y = a * (self.x ** 2) + eps
        return y

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # If training with random grid, resample coordinates per item
        if self.grid_type == 'random' and getattr(self, 'is_train', False):
            # Resample a fresh, sorted grid for this item
            x_item = (torch.rand(self.num_points) * 20 - 10).sort().values
            # Sample function parameters and noise per item
            a = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
            eps = torch.randn(self.num_points)
            y_item = a * (x_item ** 2) + eps
            return x_item.unsqueeze(-1), (y_item / 50.).unsqueeze(-1)

        # Default: fixed grid sample
        return self.x[idx, :].unsqueeze(-1), self.dataset[idx, :].unsqueeze(-1) / 50

    def get_all_points(self):
        """
        Return a representative coordinate vector for fixed-geometry precompute.
        If grid_type is 'random', return None.
        """
        if self.grid_type == 'random':
            return None
        # (N,) coordinates for the first sample
        return self.x[0]
