from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch
from torch import Tensor, nn


class EMAWeights:

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}.")
        self.decay = decay
        self.shadow: dict[str, Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        self.backup: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                self.shadow[name] = parameter.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(
                parameter.detach(),
                alpha=1.0 - self.decay,
            )

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        named_parameters = dict(model.named_parameters())
        for name, parameter in named_parameters.items():
            shadow_parameter = self.shadow.get(name)
            if shadow_parameter is None:
                continue
            parameter.data.copy_(shadow_parameter.data.to(parameter.device))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        named_parameters = dict(model.named_parameters())
        for name, parameter in named_parameters.items():
            backup_parameter = self.backup.get(name)
            if backup_parameter is None:
                continue
            parameter.data.copy_(backup_parameter.data.to(parameter.device))
        self.backup = {}

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "shadow": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.shadow.items()
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        shadow_state = state_dict.get("shadow")
        if not isinstance(shadow_state, Mapping):
            raise ValueError("EMA state is missing `shadow`.")
        self.shadow = {
            str(name): tensor.detach().clone()
            for name, tensor in shadow_state.items()
            if isinstance(tensor, Tensor)
        }
