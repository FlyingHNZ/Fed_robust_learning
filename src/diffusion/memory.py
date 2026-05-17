from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class PrototypeEntry:
    feature: Tensor
    count: int
    round_idx: int


class PrototypeMemoryBank:

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        max_entries_per_class: int,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
        if max_entries_per_class <= 0:
            raise ValueError(
                f"max_entries_per_class must be positive, got {max_entries_per_class}.",
            )

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_entries_per_class = max_entries_per_class
        self._storage: dict[int, deque[PrototypeEntry]] = {
            class_id: deque(maxlen=max_entries_per_class)
            for class_id in range(num_classes)
        }

    def update(
        self,
        local_prototypes_list: Sequence[Mapping[int, Tensor]],
        local_sample_counts_list: Sequence[Mapping[int, int]],
        round_idx: int,
    ) -> None:
        if len(local_prototypes_list) != len(local_sample_counts_list):
            raise ValueError(
                "local_prototypes_list and local_sample_counts_list must have the same length.",
            )

        for local_prototypes, sample_counts in zip(
            local_prototypes_list,
            local_sample_counts_list,
        ):
            for class_id, prototype in local_prototypes.items():
                class_count = int(sample_counts.get(class_id, 0))
                if class_count <= 0:
                    continue
                if prototype.numel() != self.feature_dim:
                    raise ValueError(
                        "Prototype feature dimension mismatch. "
                        f"Expected {self.feature_dim}, got {prototype.numel()}.",
                    )
                self._storage[class_id].append(
                    PrototypeEntry(
                        feature=prototype.detach().cpu().clone().reshape(self.feature_dim),
                        count=class_count,
                        round_idx=round_idx,
                    ),
                )

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        current_round: int,
        freshness_decay: float,
    ) -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        available_classes = self.available_classes()
        if len(available_classes) == 0:
            raise ValueError("Prototype memory bank is empty.")

        class_weights = torch.tensor(
            [
                self._class_weight(
                    class_id=class_id,
                    current_round=current_round,
                    freshness_decay=freshness_decay,
                )
                for class_id in available_classes
            ],
            device=device,
            dtype=torch.float32,
        )
        if torch.all(class_weights <= 0):
            class_weights = torch.ones_like(class_weights)
        class_tensor = torch.tensor(available_classes, device=device, dtype=torch.long)
        sampled_class_positions = torch.multinomial(
            class_weights,
            num_samples=batch_size,
            replacement=True,
        )
        sampled_labels = class_tensor[sampled_class_positions]

        sampled_features: list[Tensor] = []
        for label in sampled_labels.tolist():
            entries = list(self._storage[label])
            weights = torch.tensor(
                [
                    max(entry.count, 1)
                    * self._freshness_weight(
                        entry_round=entry.round_idx,
                        current_round=current_round,
                        freshness_decay=freshness_decay,
                    )
                    for entry in entries
                ],
                dtype=torch.float32,
            )
            if torch.all(weights <= 0):
                weights = torch.ones_like(weights)
            sampled_entry_index = int(torch.multinomial(weights, num_samples=1).item())
            sampled_features.append(entries[sampled_entry_index].feature)

        batch_features = torch.stack(sampled_features, dim=0).to(
            device=device,
            dtype=torch.float32,
        )
        return batch_features, sampled_labels

    def available_classes(self) -> list[int]:
        return [
            class_id
            for class_id, entries in self._storage.items()
            if len(entries) > 0
        ]

    def state_dict(self) -> dict[str, object]:
        return {
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "max_entries_per_class": self.max_entries_per_class,
            "storage": {
                class_id: [
                    {
                        "feature": entry.feature.detach().cpu().clone(),
                        "count": entry.count,
                        "round_idx": entry.round_idx,
                    }
                    for entry in entries
                ]
                for class_id, entries in self._storage.items()
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        storage_state = state_dict.get("storage")
        if not isinstance(storage_state, Mapping):
            raise ValueError("PrototypeMemoryBank state is missing `storage`.")

        self._storage = {
            class_id: deque(maxlen=self.max_entries_per_class)
            for class_id in range(self.num_classes)
        }

        for raw_class_id, entries in storage_state.items():
            class_id = int(raw_class_id)
            if not isinstance(entries, Sequence):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                feature = entry.get("feature")
                count = int(entry.get("count", 1))
                if not isinstance(feature, Tensor):
                    continue
                self._storage[class_id].append(
                    PrototypeEntry(
                        feature=feature.detach().cpu().clone().reshape(self.feature_dim),
                        count=count,
                        round_idx=int(entry.get("round_idx", 0)),
                    ),
                )

    def _class_weight(
        self,
        class_id: int,
        current_round: int,
        freshness_decay: float,
    ) -> float:
        return sum(
            max(entry.count, 1)
            * self._freshness_weight(
                entry_round=entry.round_idx,
                current_round=current_round,
                freshness_decay=freshness_decay,
            )
            for entry in self._storage[class_id]
        )

    @staticmethod
    def _freshness_weight(
        entry_round: int,
        current_round: int,
        freshness_decay: float,
    ) -> float:
        age = max(current_round - entry_round, 0)
        return float(torch.exp(torch.tensor(-freshness_decay * age)).item())
