from __future__ import annotations

import torch
import torchattacks
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.data.dataset import CIFAR100_MEAN, CIFAR100_STD
from src.models import SplitResNet18
from src.utils.config import FedCDPConfig


class _LogitsOnlyModel(nn.Module):

    def __init__(self, model: SplitResNet18) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        _, logits = self.model(x)
        return logits


def evaluate_robustness(
    global_model: SplitResNet18,
    test_loader: DataLoader,
    config: FedCDPConfig,
) -> dict[str, float]:
    global_model.eval()
    attack_model = _LogitsOnlyModel(global_model).to(config.device)
    attack_model.eval()

    atk_pgd = torchattacks.PGD(
        attack_model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
    )
    atk_aa = torchattacks.AutoAttack(
        attack_model,
        norm="Linf",
        eps=8 / 255,
        version="standard",
    )
    atk_cw = torchattacks.CW(
        attack_model,
        c=1.0,
        kappa=0.0,
        steps=config.cw_steps,
        lr=0.01,
    )
    atk_square = torchattacks.Square(
        attack_model,
        norm="Linf",
        eps=8 / 255,
        n_queries=config.square_queries,
        n_restarts=1,
    )
    atk_pgd.set_normalization_used(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    atk_aa.set_normalization_used(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    atk_cw.set_normalization_used(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    atk_square.set_normalization_used(mean=CIFAR100_MEAN, std=CIFAR100_STD)

    standard_correct = 0
    pgd_correct = 0
    aa_correct = 0
    cw_correct = 0
    square_correct = 0
    total_samples = 0

    for batch_inputs, batch_labels in test_loader:
        batch_inputs = batch_inputs.to(config.device, non_blocking=True)
        batch_labels = batch_labels.to(config.device, dtype=torch.long, non_blocking=True)

        with torch.no_grad():
            standard_logits = attack_model(batch_inputs)
            standard_predictions = torch.argmax(standard_logits, dim=1)
            standard_correct += int((standard_predictions == batch_labels).sum().item())

        adv_inputs_pgd = atk_pgd(batch_inputs, batch_labels)
        with torch.no_grad():
            pgd_logits = attack_model(adv_inputs_pgd)
            pgd_predictions = torch.argmax(pgd_logits, dim=1)
            pgd_correct += int((pgd_predictions == batch_labels).sum().item())

        adv_inputs_aa = atk_aa(batch_inputs, batch_labels)
        with torch.no_grad():
            aa_logits = attack_model(adv_inputs_aa)
            aa_predictions = torch.argmax(aa_logits, dim=1)
            aa_correct += int((aa_predictions == batch_labels).sum().item())

        adv_inputs_cw = atk_cw(batch_inputs, batch_labels)
        with torch.no_grad():
            cw_logits = attack_model(adv_inputs_cw)
            cw_predictions = torch.argmax(cw_logits, dim=1)
            cw_correct += int((cw_predictions == batch_labels).sum().item())

        adv_inputs_square = atk_square(batch_inputs, batch_labels)
        with torch.no_grad():
            square_logits = attack_model(adv_inputs_square)
            square_predictions = torch.argmax(square_logits, dim=1)
            square_correct += int((square_predictions == batch_labels).sum().item())

        total_samples += int(batch_labels.size(0))

    if total_samples == 0:
        return {
            "standard_accuracy_eval": 0.0,
            "pgd10_robust_accuracy": 0.0,
            "autoattack_robust_accuracy": 0.0,
            "cw_l2_robust_accuracy": 0.0,
            "square_robust_accuracy": 0.0,
        }

    return {
        "standard_accuracy_eval": standard_correct / total_samples,
        "pgd10_robust_accuracy": pgd_correct / total_samples,
        "autoattack_robust_accuracy": aa_correct / total_samples,
        "cw_l2_robust_accuracy": cw_correct / total_samples,
        "square_robust_accuracy": square_correct / total_samples,
    }
