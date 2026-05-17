from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models import SplitResNet18
from src.utils.config import FedCDPConfig


ModelState = OrderedDict[str, Tensor]
PrototypeDict = dict[int, Tensor]
ClassCountDict = dict[int, int]


class FedCDPClient:

    def __init__(
        self,
        client_id: int,
        data_loader: DataLoader,
        config: FedCDPConfig,
        model: SplitResNet18,
    ) -> None:
        self.client_id = client_id
        self.data_loader = data_loader
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.last_train_metrics: dict[str, float] = {
            "avg_total_loss": 0.0,
            "avg_local_loss": 0.0,
            "avg_proto_loss": 0.0,
            "avg_gen_loss": 0.0,
        }

    def local_train(
        self,
        global_model_state: Mapping[str, Tensor],
        global_prototypes: PrototypeDict,
        synthetic_features: Tensor | None,
        synthetic_labels: Tensor | None,
        alpha: float,
        beta: float,
    ) -> tuple[ModelState, PrototypeDict, ClassCountDict]:
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}.")

        self.model.load_state_dict(OrderedDict(global_model_state), strict=True)
        self.model.train()

        optimizer = Adam(self.model.parameters(), lr=self.config.local_lr)
        prototype_targets = {
            class_id: prototype.detach().to(self.device)
            for class_id, prototype in global_prototypes.items()
        }
        if synthetic_features is None:
            synthetic_features = torch.empty(
                0,
                self.config.feature_dim,
                device=self.device,
            )
        else:
            synthetic_features = synthetic_features.detach().to(self.device)

        if synthetic_labels is None:
            synthetic_labels = torch.empty(0, device=self.device, dtype=torch.long)
        else:
            synthetic_labels = synthetic_labels.detach().to(self.device, dtype=torch.long)

        total_loss_sum = 0.0
        local_loss_sum = 0.0
        proto_loss_sum = 0.0
        gen_loss_sum = 0.0
        num_batches = 0

        for _ in range(self.config.local_epochs):
            for batch_inputs, batch_labels in self.data_loader:
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, dtype=torch.long, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                features, logits = self.model(batch_inputs)

                local_loss = F.cross_entropy(logits, batch_labels)
                proto_loss = self._compute_proto_loss(
                    features=features,
                    labels=batch_labels,
                    global_prototypes=prototype_targets,
                )
                gen_loss = self._compute_gen_loss(
                    synthetic_features=synthetic_features,
                    synthetic_labels=synthetic_labels,
                )

                loss = local_loss + alpha * proto_loss + beta * gen_loss
                loss.backward()
                optimizer.step()

                total_loss_sum += float(loss.detach().item())
                local_loss_sum += float(local_loss.detach().item())
                proto_loss_sum += float(proto_loss.detach().item())
                gen_loss_sum += float(gen_loss.detach().item())
                num_batches += 1

        if num_batches > 0:
            self.last_train_metrics = {
                "avg_total_loss": total_loss_sum / num_batches,
                "avg_local_loss": local_loss_sum / num_batches,
                "avg_proto_loss": proto_loss_sum / num_batches,
                "avg_gen_loss": gen_loss_sum / num_batches,
            }
        else:
            self.last_train_metrics = {
                "avg_total_loss": 0.0,
                "avg_local_loss": 0.0,
                "avg_proto_loss": 0.0,
                "avg_gen_loss": 0.0,
            }

        local_prototypes, sample_counts = self.compute_local_prototypes()
        uploaded_prototypes = self.privatize_local_prototypes(local_prototypes)
        updated_state = OrderedDict(
            (name, parameter.detach().cpu().clone())
            for name, parameter in self.model.state_dict().items()
        )
        return updated_state, uploaded_prototypes, sample_counts

    def compute_local_prototypes(self) -> tuple[PrototypeDict, ClassCountDict]:
        feature_sums: dict[int, Tensor] = {}
        class_counts: defaultdict[int, int] = defaultdict(int)

        self.model.eval()
        with torch.no_grad():
            for batch_inputs, batch_labels in self.data_loader:
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, dtype=torch.long, non_blocking=True)

                features, _ = self.model(batch_inputs)
                for feature, label in zip(features, batch_labels):
                    feature = self._clip_feature_norm(feature.detach())
                    class_id = int(label.item())
                    if class_id not in feature_sums:
                        feature_sums[class_id] = torch.zeros(
                            self.config.feature_dim,
                            device=self.device,
                            dtype=feature.dtype,
                        )
                    feature_sums[class_id] += feature.detach()
                    class_counts[class_id] += 1

        local_prototypes = {
            class_id: (feature_sum / class_counts[class_id]).detach().cpu().clone()
            for class_id, feature_sum in feature_sums.items()
        }
        return local_prototypes, dict(class_counts)

    def privatize_local_prototypes(
        self,
        local_prototypes: PrototypeDict,
    ) -> PrototypeDict:
        sigma = (
            self.config.prototype_clip_norm
            * math.sqrt(2.0 * math.log(1.25 / self.config.dp_delta))
            / self.config.dp_epsilon
        )
        uploaded_prototypes: PrototypeDict = {}
        for class_id, prototype in local_prototypes.items():
            clipped_prototype = self._clip_feature_norm(prototype.to(self.device))
            noise = torch.randn_like(clipped_prototype) * sigma
            uploaded_prototypes[class_id] = (clipped_prototype + noise).detach().cpu().clone()
        return uploaded_prototypes

    def _compute_proto_loss(
        self,
        features: Tensor,
        labels: Tensor,
        global_prototypes: PrototypeDict,
    ) -> Tensor:
        per_sample_loss = torch.zeros(features.size(0), device=features.device, dtype=features.dtype)
        for sample_index, label in enumerate(labels):
            class_id = int(label.item())
            prototype = global_prototypes.get(class_id)
            if prototype is None:
                continue
            per_sample_loss[sample_index] = F.mse_loss(
                features[sample_index],
                prototype,
                reduction="mean",
            )
        return per_sample_loss.mean()

    def _compute_gen_loss(
        self,
        synthetic_features: Tensor,
        synthetic_labels: Tensor,
    ) -> Tensor:
        if synthetic_features.numel() == 0 or synthetic_labels.numel() == 0:
            return torch.zeros((), device=self.device)

        synthetic_logits = self.model.classifier(synthetic_features)
        return F.cross_entropy(synthetic_logits, synthetic_labels)

    def _clip_feature_norm(self, feature: Tensor) -> Tensor:
        feature_norm = torch.norm(feature, p=2)
        if feature_norm <= self.config.prototype_clip_norm:
            return feature
        scale = self.config.prototype_clip_norm / feature_norm.clamp(min=1e-12)
        return feature * scale
