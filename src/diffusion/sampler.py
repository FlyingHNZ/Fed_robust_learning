from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from src.diffusion.mlp_diffusion import FeatureDenoiser
from src.utils.config import FedCDPConfig


PrototypeDict = dict[int, Tensor]


class ContrastiveGuidedDDIM:

    def __init__(
        self,
        denoiser: FeatureDenoiser,
        variance_schedule: Sequence[float] | Tensor,
        S: int = 50,
        config: FedCDPConfig | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.device = next(denoiser.parameters()).device
        self.config = config
        self.variance_schedule = torch.as_tensor(
            variance_schedule,
            dtype=torch.float32,
            device=self.device,
        )
        if self.variance_schedule.ndim != 1 or self.variance_schedule.numel() == 0:
            raise ValueError("variance_schedule must be a non-empty 1D sequence.")
        if S <= 0:
            raise ValueError(f"S must be positive, got {S}.")

        self.S = min(S, int(self.variance_schedule.numel()))
        self.alphas = 1.0 - self.variance_schedule
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def contrastive_guidance_gradient(
        self,
        z_s: Tensor,
        target_labels: Tensor,
        global_prototypes: Mapping[int, Tensor],
        tau: float,
    ) -> Tensor:
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}.")
        if len(global_prototypes) == 0:
            raise ValueError("global_prototypes must not be empty.")

        z_s = z_s.requires_grad_(True)
        z_norm = F.normalize(z_s, p=2, dim=-1)
        available_classes = sorted(global_prototypes.keys())
        class_to_position = {
            class_id: position for position, class_id in enumerate(available_classes)
        }
        missing_labels = sorted(
            {
                int(label.item())
                for label in target_labels
                if int(label.item()) not in class_to_position
            },
        )
        if len(missing_labels) > 0:
            raise ValueError(
                "All target labels must be present in global_prototypes. "
                f"Missing labels: {missing_labels}.",
            )

        P = torch.stack(
            [
                global_prototypes[class_id].detach().to(
                    device=z_s.device,
                    dtype=z_s.dtype,
                )
                for class_id in available_classes
            ],
            dim=0,
        )
        P = F.normalize(P, p=2, dim=-1)
        sim_matrix = torch.matmul(z_norm, P.T)
        target_positions = torch.tensor(
            [class_to_position[int(label.item())] for label in target_labels],
            device=z_s.device,
            dtype=torch.long,
        )
        sim_pos = sim_matrix.gather(
            dim=1,
            index=target_positions.view(-1, 1),
        ).squeeze(1)

        logits = sim_matrix / tau
        log_denom = torch.logsumexp(logits, dim=1)
        L_con = -(sim_pos / tau - log_denom)
        grad_con = torch.autograd.grad(
            outputs=L_con.sum(),
            inputs=z_s,
        )[0]
        return grad_con

    def sample(
        self,
        M: int,
        global_prototypes: Mapping[int, Tensor],
        target_labels: Tensor,
        config: FedCDPConfig | None = None,
    ) -> Tensor:
        effective_config = config if config is not None else self.config
        if effective_config is None:
            raise ValueError("A FedCDPConfig must be provided either at init time or sample time.")
        if M <= 0:
            raise ValueError(f"M must be positive, got {M}.")
        if target_labels.numel() != M:
            raise ValueError(
                f"target_labels must contain exactly M={M} labels, got {target_labels.numel()}.",
            )

        self.denoiser.eval()
        target_labels = target_labels.to(self.device, dtype=torch.long)
        z_s = torch.randn(M, effective_config.feature_dim, device=self.device)

        seq = torch.linspace(
            0,
            self.variance_schedule.numel() - 1,
            steps=self.S,
            device=self.device,
        ).long()
        seq_next = torch.cat(
            [
                torch.full((1,), -1, device=self.device, dtype=torch.long),
                seq[:-1],
            ],
            dim=0,
        )

        for current_step, next_step in zip(reversed(seq.tolist()), reversed(seq_next.tolist())):
            current_index = int(current_step)
            next_index = int(next_step)

            alpha_t = self.alphas_cumprod[current_index]
            alpha_next = (
                self.alphas_cumprod[next_index]
                if next_index >= 0
                else torch.tensor(1.0, device=self.device, dtype=self.alphas_cumprod.dtype)
            )

            timestep = torch.full(
                (M,),
                fill_value=float(current_index),
                device=self.device,
                dtype=torch.float32,
            )

            with torch.no_grad():
                eps_uncond = self.denoiser(
                    z_s,
                    timestep,
                    labels=None,
                    force_unconditional=True,
                )
                eps_cond = self.denoiser(
                    z_s,
                    timestep,
                    labels=target_labels,
                    force_unconditional=False,
                )
                eps = eps_uncond + effective_config.cfg_guidance_scale * (
                    eps_cond - eps_uncond
                )

            grad_con = self.contrastive_guidance_gradient(
                z_s=z_s,
                target_labels=target_labels,
                global_prototypes=global_prototypes,
                tau=effective_config.contrastive_temperature,
            )
            eps_hat = eps - effective_config.guidance_lambda * grad_con

            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()
            x0_t = (z_s - eps_hat * sqrt_one_minus_alpha_t) / sqrt_alpha_t

            eta = 0.0
            c1 = eta * (
                ((1.0 - alpha_t / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha_t)).sqrt()
                if next_index >= 0
                else torch.tensor(0.0, device=self.device, dtype=self.alphas_cumprod.dtype)
            )
            c2 = ((1.0 - alpha_next) - c1**2).clamp(min=0.0).sqrt()

            z_prev = (
                alpha_next.sqrt() * x0_t
                + c1 * torch.randn_like(z_s)
                + c2 * eps_hat
            )
            z_s = z_prev.detach()

        return z_s
