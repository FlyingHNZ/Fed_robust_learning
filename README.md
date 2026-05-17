# Federated Contrastive Diffusion Prototypes for Robust Private Learning

The repo is a modular PyTorch reproduction scaffold for **Federated Contrastive Diffusion Prototypes**, a robust federated learning framework designed for severe non-IID settings.

## Environment Setup

We recommend to use an Anaconda environment.

### Install dependencies

If dependencies are not already installed, run:

```powershell
pip install -r requirements.txt
```

Dependencies listed in `requirements.txt`:

```text
torch>=2.0.0
torchvision
numpy
scipy
tqdm
wandb
torchattacks
```

## Running Training

Run the main federated training loop with:

```powershell
conda activate your environment
Set-Location your directory
python .\main.py
```

Run with the provided YAML config:

```powershell
conda activate your environment
Set-Location your directory
python .\main.py --config .\configs\fedcdp.yaml
```

## Logging

The current implementation uses `wandb` in `main.py`.

If `wandb` login is required in your environment, run:

```powershell
wandb login
```

