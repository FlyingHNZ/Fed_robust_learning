from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from src.utils.config import FedCDPConfig


ModelState = OrderedDict[str, Tensor]
PrototypeDict = dict[int, Tensor]
ClassCountDict = dict[int, int]

class FedCDPServer:

    def __init__(self, config: FedCDPConfig) -> None:
        self.config = config
        self.ema_prototypes: PrototypeDict = {}

    def aggregate_models(
        self,
        client_model_states: Sequence[Mapping[str, Tensor]],
        client_sample_counts: Sequence[int],
    ) -> ModelState:
        if len(client_model_states) == 0:
            raise ValueError("client_model_states must not be empty.")
        if len(client_model_states) != len(client_sample_counts):
            raise ValueError(
                "client_model_states and client_sample_counts must have the same length.",
            )

        total_samples = sum(client_sample_counts)
        if total_samples <= 0:
            raise ValueError(
                "The total number of client samples must be positive for FedAvg.",
            )

        first_state = client_model_states[0]
        aggregated_state: ModelState = OrderedDict()
        for parameter_name, parameter in first_state.items():
            accumulation_dtype = (
                parameter.dtype if parameter.is_floating_point() else torch.float32
            )
            aggregated_state[parameter_name] = torch.zeros_like(
                parameter,
                dtype=accumulation_dtype,
            )

        for state_dict, sample_count in zip(client_model_states, client_sample_counts):
            if sample_count < 0:
                raise ValueError(
                    f"Client sample counts must be non-negative, got {sample_count}.",
                )

            weight = sample_count / total_samples
            for parameter_name, parameter in state_dict.items():
                aggregated_state[parameter_name] += (
                    parameter.detach().clone().to(aggregated_state[parameter_name].dtype)
                    * weight
                )

        for parameter_name, parameter in first_state.items():
            if aggregated_state[parameter_name].dtype != parameter.dtype:
                aggregated_state[parameter_name] = (
                    aggregated_state[parameter_name]
                    .round()
                    .to(dtype=parameter.dtype)
                )

        return aggregated_state

    def aggregate_prototypes(
        self,
        local_prototypes_list: Sequence[Mapping[int, Tensor]],
        local_sample_counts_list: Sequence[Mapping[int, int]],
    ) -> PrototypeDict:
        if len(local_prototypes_list) != len(local_sample_counts_list):
            raise ValueError(
                "local_prototypes_list and local_sample_counts_list must have the same length.",
            )

        global_prototypes: PrototypeDict = {}
        for class_id in range(self.config.num_classes):
            weighted_sum: Tensor | None = None
            total_class_samples = 0

            for local_prototypes, sample_counts in zip(
                local_prototypes_list,
                local_sample_counts_list,
            ):
                class_count = sample_counts.get(class_id, 0)
                local_prototype = local_prototypes.get(class_id)
                if class_count <= 0 or local_prototype is None:
                    continue

                local_prototype = local_prototype.detach().cpu()
                if weighted_sum is None:
                    weighted_sum = torch.zeros_like(local_prototype)
                weighted_sum += local_prototype * class_count
                total_class_samples += class_count

            if weighted_sum is not None and total_class_samples > 0:
                current_prototype = weighted_sum / total_class_samples
                previous_ema = self.ema_prototypes.get(class_id)
                if previous_ema is None:
                    self.ema_prototypes[class_id] = current_prototype.detach().clone()
                else:
                    self.ema_prototypes[class_id] = (
                        self.config.ema_momentum * previous_ema.detach().cpu()
                        + (1.0 - self.config.ema_momentum) * current_prototype
                    )
                global_prototypes[class_id] = current_prototype.detach().clone()
            elif class_id in self.ema_prototypes:
                global_prototypes[class_id] = self.ema_prototypes[class_id].detach().clone()

        return global_prototypes

    def apply_dp_to_prototypes(
        self,
        global_prototypes: Mapping[int, Tensor],
    ) -> PrototypeDict:
        sigma = (
            1.0
            * math.sqrt(2.0 * math.log(1.25 / self.config.dp_delta))
            / self.config.dp_epsilon
        )

        dp_protected_prototypes: PrototypeDict = {}
        for class_id, prototype in global_prototypes.items():
            base_prototype = prototype.detach().clone()
            noise = torch.randn_like(base_prototype) * sigma
            noisy_prototype = base_prototype + noise
            dp_protected_prototypes[class_id] = F.normalize(
                noisy_prototype,
                p=2,
                dim=-1,
            )

        return dp_protected_prototypes
