from __future__ import annotations

import math

import torch
from torch import Tensor

from src.utils.config import FedCDPConfig


def build_variance_schedule(config: FedCDPConfig) -> Tensor:
    if config.diffusion_schedule == "linear":
        return _linear_schedule(
            num_steps=config.diffusion_steps,
            beta_start=config.diffusion_schedule_start,
            beta_end=config.diffusion_schedule_end,
        )
    if config.diffusion_schedule == "cosine":
        return _cosine_schedule(
            num_steps=config.diffusion_steps,
            s=config.diffusion_schedule_s,
        )
    if config.diffusion_schedule == "sigmoid":
        return _sigmoid_schedule(
            num_steps=config.diffusion_steps,
            beta_start=config.diffusion_schedule_start,
            beta_end=config.diffusion_schedule_end,
        )
    raise ValueError(f"Unsupported diffusion schedule: {config.diffusion_schedule}.")


def _linear_schedule(
    num_steps: int,
    beta_start: float,
    beta_end: float,
) -> Tensor:
    return torch.linspace(beta_start, beta_end, steps=num_steps, dtype=torch.float32)


def _cosine_schedule(num_steps: int, s: float) -> Tensor:
    timesteps = torch.arange(num_steps + 1, dtype=torch.float32)
    alpha_bar = torch.cos(((timesteps / num_steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(min=1e-5, max=0.999)


def _sigmoid_schedule(
    num_steps: int,
    beta_start: float,
    beta_end: float,
) -> Tensor:
    timesteps = torch.linspace(-6.0, 6.0, steps=num_steps, dtype=torch.float32)
    sig = torch.sigmoid(timesteps)
    sig = (sig - sig.min()) / (sig.max() - sig.min())
    return beta_start + (beta_end - beta_start) * sig
