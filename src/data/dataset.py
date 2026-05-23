from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch.utils.data import DataLoader

from src.utils.config import FedCDPConfig

@dataclass(slots=True)
class AttackNormalization:
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(slots=True)
class FederatedDataBundle:
    client_loaders: list[DataLoader]
    test_loader: DataLoader
    attack_normalization: AttackNormalization | None = None


class FederatedDataProvider(Protocol):
    def __call__(self, config: FedCDPConfig) -> FederatedDataBundle:
        """Build the full federated data bundle for one experiment run."""


_ACTIVE_PROVIDER: FederatedDataProvider | None = None


def register_federated_data_provider(provider: FederatedDataProvider) -> None:
    global _ACTIVE_PROVIDER
    _ACTIVE_PROVIDER = provider


def clear_federated_data_provider() -> None:
    global _ACTIVE_PROVIDER
    _ACTIVE_PROVIDER = None


def get_federated_data_bundle(config: FedCDPConfig) -> FederatedDataBundle:
    if _ACTIVE_PROVIDER is None:
        raise NotImplementedError(
            "No federated data provider is registered. "
            "Implement your own dataset preparation and call "
            "`register_federated_data_provider(...)` before training or evaluation.",
        )

    bundle = _ACTIVE_PROVIDER(config)
    _validate_data_bundle(bundle=bundle, config=config)
    return bundle


def get_federated_dataloaders(config: FedCDPConfig) -> tuple[list[DataLoader], DataLoader]:
    bundle = get_federated_data_bundle(config)
    return bundle.client_loaders, bundle.test_loader


def _validate_data_bundle(bundle: FederatedDataBundle, config: FedCDPConfig) -> None:
    if len(bundle.client_loaders) == 0:
        raise ValueError("The registered data provider returned no client dataloaders.")
    if len(bundle.client_loaders) < config.client_selection_size:
        raise ValueError(
            "The registered data provider returned fewer client dataloaders than "
            f"`config.client_selection_size` ({len(bundle.client_loaders)} < "
            f"{config.client_selection_size}).",
        )
    if bundle.attack_normalization is not None:
        if len(bundle.attack_normalization.mean) != len(bundle.attack_normalization.std):
            raise ValueError(
                "attack_normalization.mean and attack_normalization.std must have the same length.",
            )
