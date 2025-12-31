# HDM for Quadratic Dataset

This is a cleaned version of the Hilbert Diffusion Model (HDM) code, focused only on 1D function generation using the Quadratic dataset.

## What was removed
- All 2D image-related code and models (UNet, DDPM, NCSNPP)
- Image datasets (CIFAR10, MNIST, LSUN, FFHQ, AFHQ)
- Image evaluation metrics (FID, PRDC)
- Unnecessary dependencies (PIL, torchvision, blobfile, clean-fid, kornia, scikit-image)

## Remaining structure
```
├── configs/
│   └── hdm_quadratic_fno.yml    # Configuration for Quadratic dataset
├── datasets/
│   ├── __init__.py              # Dataset loader
│   ├── quadratic.py             # Quadratic dataset implementation
│   ├── melbourne.py             # Melbourne dataset (optional)
│   └── gridwatch.py             # Gridwatch dataset (optional)
├── models/
│   ├── __init__.py
│   ├── fno.py                   # Fourier Neural Operator
│   ├── fno_block.py             # FNO building blocks
│   ├── temp_fno.py              # Temporal FNO
│   ├── temp_fno_block.py        # Temporal FNO blocks
│   ├── temp_mlp.py              # Temporal MLP
│   └── mlp.py                   # MLP model
├── functions/
│   ├── __init__.py
│   ├── sde.py                   # VPSDE1D for 1D data
│   ├── loss.py                  # Loss functions
│   ├── sampler.py               # Sampling algorithms
│   ├── ihdm.py                  # Inverse HDM
│   └── utils.py                 # Utility functions
├── runner/
│   ├── __init__.py
│   └── hilbert_runner.py        # HilbertDiffusion class for 1D
├── evaluate/
│   ├── __init__.py
│   └── power.py                 # Power evaluation metrics
├── main.py                      # Main entry point
└── requirements.txt             # Python dependencies
```

## Installation

### Local Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Installation
```bash
# Run the installation script
!bash colab_install.sh

# Or install manually
!pip install einops matplotlib numpy pandas scipy tensorboard tensorly tensorly-torch tqdm transformers scikit-learn glob2
```

## Usage

### Training
```bash
python main.py --config hdm_quadratic_fno.yml --doc quadratic_experiment
```

### Sampling
```bash
python main.py --config hdm_quadratic_fno.yml --doc quadratic_experiment --sample
```

### Configuration
The main configuration file is `configs/hdm_quadratic_fno.yml`. Key parameters:
- `data.num_data`: Number of training samples (default: 1000)
- `data.dimension`: Number of points per function (default: 100)
- `training.epochs`: Training epochs (default: 2000)
- `model.type`: Model architecture (FNO for 1D data)

## Quadratic Dataset
The Quadratic dataset generates 1D quadratic functions of the form:
```
y = a * x² + ε
```
where:
- `a` is randomly chosen from {-1, 1}
- `x` ranges from -10 to 10
- `ε` is Gaussian noise with std=1
- Output is normalized by dividing by 50