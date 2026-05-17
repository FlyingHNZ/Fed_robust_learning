from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

from src.diffusion.ema import EMAWeights
from src.diffusion.memory import PrototypeMemoryBank
from src.diffusion.mlp_diffusion import FeatureDenoiser
from src.utils.config import FedCDPConfig


class FeatureDiffusionTrainer:

    def __init__(
        self,
        denoiser: FeatureDenoiser,
        variance_schedule: Sequence[float] | Tensor,
        config: FedCDPConfig,
    ) -> None:
        self.denoiser = denoiser
        self.config = config
        self.device = next(denoiser.parameters()).device
        self.variance_schedule = torch.as_tensor(
            variance_schedule,
            dtype=torch.float32,
            device=self.device,
        )
        if self.variance_schedule.ndim != 1 or self.variance_schedule.numel() == 0:
            raise ValueError("variance_schedule must be a non-empty 1D tensor.")

        self.alphas = 1.0 - self.variance_schedule
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.optimizer = AdamW(
            self.denoiser.parameters(),
            lr=config.denoiser_lr,
            weight_decay=config.denoiser_weight_decay,
        )
        self.ema_helper = EMAWeights(
            model=self.denoiser,
            decay=config.denoiser_ema_decay,
        )
        self.prototype_memory = PrototypeMemoryBank(
            num_classes=config.num_classes,
            feature_dim=config.feature_dim,
            max_entries_per_class=config.prototype_memory_size_per_class,
        )

    def train_round(
        self,
        local_prototypes_list: Sequence[Mapping[int, Tensor]],
        local_sample_counts_list: Sequence[Mapping[int, int]],
        global_prototypes: Mapping[int, Tensor],
        classifier: nn.Linear,
        round_idx: int,
    ) -> dict[str, float]:
        if len(local_prototypes_list) != len(local_sample_counts_list):
            raise ValueError(
                "local_prototypes_list and local_sample_counts_list must have the same length.",
            )

        self.prototype_memory.update(
            local_prototypes_list=local_prototypes_list,
            local_sample_counts_list=local_sample_counts_list,
            round_idx=round_idx,
        )
        if len(global_prototypes) == 0 or len(self.prototype_memory.available_classes()) == 0:
            return {
                "avg_total_loss": 0.0,
                "avg_mse_loss": 0.0,
                "avg_contrastive_loss": 0.0,
                "avg_reconstruction_loss": 0.0,
                "avg_classifier_loss": 0.0,
            }

        prototype_bank, class_ids = self._stack_prototypes(global_prototypes)
        class_to_position = {
            class_id: position for position, class_id in enumerate(class_ids)
        }
        self.denoiser.train()
        classifier = classifier.to(self.device)
        classifier.eval()

        total_loss_sum = 0.0
        mse_loss_sum = 0.0
        contrastive_loss_sum = 0.0
        reconstruction_loss_sum = 0.0
        classifier_loss_sum = 0.0
        cfg_dropout_count = 0.0

        for _ in range(self.config.denoiser_steps_per_round):
            clean_features, class_labels = self.prototype_memory.sample_batch(
                batch_size=self.config.denoiser_batch_size,
                device=self.device,
                current_round=round_idx,
                freshness_decay=self.config.replay_freshness_decay,
            )
            target_positions = torch.tensor(
                [class_to_position[int(label)] for label in class_labels.tolist()],
                device=self.device,
                dtype=torch.long,
            )
            timestep = self._sample_antithetic_timesteps(clean_features.size(0))
            noise = torch.randn_like(clean_features)
            noisy_features = self.q_sample(clean_features, timestep, noise)
            conditioned_labels, dropped_count = self._apply_cfg_dropout(class_labels)

            predicted_noise = self.denoiser(
                noisy_features,
                timestep.float(),
                labels=conditioned_labels,
            )
            mse_loss = F.mse_loss(predicted_noise, noise)

            predicted_x0 = self.predict_x0(
                noisy_features=noisy_features,
                timestep=timestep,
                predicted_noise=predicted_noise,
            )
            reconstruction_loss = F.mse_loss(predicted_x0, clean_features)
            contrastive_loss = self._prototype_contrastive_loss(
                predicted_x0=predicted_x0,
                prototype_bank=prototype_bank,
                target_positions=target_positions,
            )
            classifier_loss = self._classifier_consistency_loss(
                predicted_x0=predicted_x0,
                labels=class_labels,
                classifier=classifier,
            )

            total_loss = (
                self.config.denoiser_mse_weight * mse_loss
                + self.config.denoiser_contrastive_weight * contrastive_loss
                + self.config.denoiser_reconstruction_weight * reconstruction_loss
                + self.config.denoiser_classifier_weight * classifier_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.denoiser.parameters(),
                max_norm=self.config.denoiser_grad_clip,
            )
            self.optimizer.step()
            self.ema_helper.update(self.denoiser)

            total_loss_sum += float(total_loss.detach().item())
            mse_loss_sum += float(mse_loss.detach().item())
            contrastive_loss_sum += float(contrastive_loss.detach().item())
            reconstruction_loss_sum += float(reconstruction_loss.detach().item())
            classifier_loss_sum += float(classifier_loss.detach().item())
            cfg_dropout_count += float(dropped_count)

        step_count = max(self.config.denoiser_steps_per_round, 1)
        return {
            "avg_total_loss": total_loss_sum / step_count,
            "avg_mse_loss": mse_loss_sum / step_count,
            "avg_contrastive_loss": contrastive_loss_sum / step_count,
            "avg_reconstruction_loss": reconstruction_loss_sum / step_count,
            "avg_classifier_loss": classifier_loss_sum / step_count,
            "avg_cfg_dropout_count": cfg_dropout_count / step_count,
        }

    def q_sample(self, clean_features: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        alpha_bar = self.alphas_cumprod[timestep].unsqueeze(1)
        return alpha_bar.sqrt() * clean_features + (1.0 - alpha_bar).sqrt() * noise

    def predict_x0(
        self,
        noisy_features: Tensor,
        timestep: Tensor,
        predicted_noise: Tensor,
    ) -> Tensor:
        alpha_bar = self.alphas_cumprod[timestep].unsqueeze(1)
        return (
            noisy_features - (1.0 - alpha_bar).sqrt() * predicted_noise
        ) / alpha_bar.sqrt().clamp(min=1e-6)

    def _stack_prototypes(
        self,
        global_prototypes: Mapping[int, Tensor],
    ) -> tuple[Tensor, list[int]]:
        class_ids = sorted(global_prototypes.keys())
        prototype_bank = torch.stack(
            [
                global_prototypes[class_id].detach().to(
                    device=self.device,
                    dtype=torch.float32,
                )
                for class_id in class_ids
            ],
            dim=0,
        )
        return prototype_bank, class_ids

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema_helper.state_dict(),
            "prototype_memory": self.prototype_memory.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        optimizer_state = state_dict.get("optimizer")
        if isinstance(optimizer_state, Mapping):
            self.optimizer.load_state_dict(optimizer_state)

        ema_state = state_dict.get("ema")
        if isinstance(ema_state, Mapping):
            self.ema_helper.load_state_dict(ema_state)

        prototype_memory_state = state_dict.get("prototype_memory")
        if isinstance(prototype_memory_state, Mapping):
            self.prototype_memory.load_state_dict(prototype_memory_state)

    @contextmanager
    def use_ema_weights(self) -> object:
        with self.ema_helper.apply_to(self.denoiser):
            yield

    def _prototype_contrastive_loss(
        self,
        predicted_x0: Tensor,
        prototype_bank: Tensor,
        target_positions: Tensor,
    ) -> Tensor:
        if prototype_bank.size(0) <= 1 or self.config.denoiser_contrastive_weight == 0.0:
            return torch.zeros((), device=self.device)

        predicted_x0 = F.normalize(predicted_x0, p=2, dim=-1)
        prototype_bank = F.normalize(prototype_bank, p=2, dim=-1)
        logits = torch.matmul(predicted_x0, prototype_bank.T) / self.config.contrastive_temperature
        return F.cross_entropy(logits, target_positions)

    def _classifier_consistency_loss(
        self,
        predicted_x0: Tensor,
        labels: Tensor,
        classifier: nn.Linear,
    ) -> Tensor:
        with torch.no_grad():
            target_weight = classifier.weight.detach().clone()
            target_bias = (
                classifier.bias.detach().clone()
                if classifier.bias is not None
                else None
            )

        logits = F.linear(predicted_x0, target_weight, target_bias)
        return F.cross_entropy(logits, labels)

    def _sample_antithetic_timesteps(self, batch_size: int) -> Tensor:
        half = batch_size // 2 + batch_size % 2
        timestep = torch.randint(
            low=0,
            high=self.variance_schedule.numel(),
            size=(half,),
            device=self.device,
        )
        mirrored_timestep = self.variance_schedule.numel() - timestep - 1
        timestep = torch.cat([timestep, mirrored_timestep], dim=0)[:batch_size]
        return timestep.to(dtype=torch.long)

    def _apply_cfg_dropout(self, class_labels: Tensor) -> tuple[Tensor, int]:
        if self.config.cfg_dropout_prob <= 0.0:
            return class_labels, 0

        dropout_mask = torch.rand(
            class_labels.size(0),
            device=class_labels.device,
        ) < self.config.cfg_dropout_prob
        conditioned_labels = class_labels.clone()
        conditioned_labels[dropout_mask] = self.denoiser.null_class_index
        return conditioned_labels, int(dropout_mask.sum().item())
