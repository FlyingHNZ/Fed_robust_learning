from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import wandb
from torch import Tensor
from torch.utils.data import DataLoader

from src.data import get_cifar100_federated
from src.diffusion import (
    ContrastiveGuidedDDIM,
    FeatureDiffusionTrainer,
    FeatureDenoiser,
    build_variance_schedule,
    filter_synthetic_features,
)
from src.fl_core import FedCDPClient, FedCDPServer
from src.models import SplitResNet18
from src.utils.config import FedCDPConfig
from src.utils.config_loader import load_config, parse_override_items
from src.utils.evaluation import evaluate_robustness


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_global_model(
    model: SplitResNet18,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, dtype=torch.long, non_blocking=True)

            _, logits = model(batch_inputs)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += int((predictions == batch_labels).sum().item())
            total_samples += int(batch_labels.size(0))

    if total_samples == 0:
        return 0.0
    return correct_predictions / total_samples

def sample_target_labels(
    num_samples: int,
    available_classes: Sequence[int],
    device: torch.device,
) -> Tensor:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if len(available_classes) == 0:
        raise ValueError("available_classes must not be empty.")

    class_tensor = torch.tensor(
        list(available_classes),
        device=device,
        dtype=torch.long,
    )
    sampled_positions = torch.randint(
        low=0,
        high=class_tensor.numel(),
        size=(num_samples,),
        device=device,
    )
    return class_tensor[sampled_positions]

def load_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint format is invalid.")
    return checkpoint


def maybe_run_robustness_eval(
    round_idx: int,
    global_model: SplitResNet18,
    test_loader: DataLoader,
    config: FedCDPConfig,
) -> dict[str, float]:
    if round_idx % config.robust_eval_interval != 0:
        return {}

    return evaluate_robustness(
        global_model=global_model,
        test_loader=test_loader,
        config=config,
    )


def maybe_save_best_checkpoint(
    checkpoint_dir: Path,
    round_idx: int,
    config: FedCDPConfig,
    global_model: SplitResNet18,
    denoiser: FeatureDenoiser,
    diffusion_trainer: FeatureDiffusionTrainer,
    server: FedCDPServer,
    synthetic_features: Tensor | None,
    synthetic_labels: Tensor | None,
    metrics: dict[str, float],
    current_best_value: float,
) -> float:
    metric_name = config.best_checkpoint_metric
    candidate_value = metrics.get(metric_name)
    if candidate_value is None or candidate_value <= current_best_value:
        return current_best_value

    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        round_idx=round_idx,
        config=config,
        global_model=global_model,
        denoiser=denoiser,
        diffusion_trainer=diffusion_trainer,
        server=server,
        synthetic_features=synthetic_features,
        synthetic_labels=synthetic_labels,
        checkpoint_name="best.pt",
        extra_state={
            "best_metric_name": metric_name,
            "best_metric_value": candidate_value,
        },
    )
    return candidate_value

def run_training(
    config: FedCDPConfig,
    checkpoint_path: Path | None = None,
    run_name: str = "fedcdp-main-loop",
) -> None:
    set_global_seed(config.seed)

    client_loaders, test_loader = get_cifar100_federated(config)

    global_model = SplitResNet18(num_classes=config.num_classes).to(config.device)
    server = FedCDPServer(config)
    denoiser = FeatureDenoiser(
        feature_dim=config.feature_dim,
        time_emb_dim=config.denoiser_time_emb_dim,
        num_classes=config.num_classes,
    ).to(config.device)
    variance_schedule = build_variance_schedule(config)
    diffusion_trainer = FeatureDiffusionTrainer(
        denoiser=denoiser,
        variance_schedule=variance_schedule,
        config=config,
    )
    sampler = ContrastiveGuidedDDIM(
        denoiser=denoiser,
        variance_schedule=variance_schedule,
        S=config.diffusion_steps,
        config=config,
    )

    clients = [
        FedCDPClient(
            client_id=client_id,
            data_loader=client_loader,
            config=config,
            model=SplitResNet18(num_classes=config.num_classes),
        )
        for client_id, client_loader in enumerate(client_loaders)
    ]

    synthetic_features: Tensor | None = None
    synthetic_labels: Tensor | None = None
    global_prototypes: dict[int, Tensor] = {}
    checkpoint_dir = Path(config.checkpoint_dir)
    start_round = 0
    best_metric_value = float("-inf")

    if checkpoint_path is not None:
        checkpoint = load_checkpoint_state(checkpoint_path)
        global_model.load_state_dict(checkpoint["global_model_state"], strict=True)
        denoiser.load_state_dict(checkpoint["denoiser_state"], strict=True)
        trainer_state = checkpoint.get("diffusion_trainer_state")
        if isinstance(trainer_state, dict):
            diffusion_trainer.load_state_dict(trainer_state)
        elif "denoiser_optimizer_state" in checkpoint:
            diffusion_trainer.optimizer.load_state_dict(checkpoint["denoiser_optimizer_state"])
        server.ema_prototypes = {
            int(class_id): prototype.detach().cpu().clone()
            for class_id, prototype in checkpoint.get("ema_prototypes", {}).items()
        }
        global_prototypes = {
            class_id: prototype.detach().cpu().clone()
            for class_id, prototype in server.ema_prototypes.items()
        }

        stored_synthetic_features = checkpoint.get("synthetic_features")
        if isinstance(stored_synthetic_features, torch.Tensor):
            synthetic_features = stored_synthetic_features.to(config.device)

        stored_synthetic_labels = checkpoint.get("synthetic_labels")
        if isinstance(stored_synthetic_labels, torch.Tensor):
            synthetic_labels = stored_synthetic_labels.to(config.device, dtype=torch.long)

        start_round = int(checkpoint.get("round", 0))
        best_metric_value = float(
            checkpoint.get("best_metric_value", float("-inf")),
        )

    wandb.init(
        project="Fed-CDP",
        name=run_name,
        config=config.to_dict(),
    )

    try:
        for round_t in range(start_round, config.num_rounds):
            round_idx = round_t + 1
            current_alpha = min(1.0, 0.01 + (round_t / 20.0) * 0.99)
            current_beta = min(1.0, 0.01 + (round_t / 20.0) * 0.99)
            selection_size = max(1, int(config.num_clients * config.fraction_fit))

            selected_clients = random.sample(
                clients,
                min(selection_size, len(clients)),
            )
            global_model_state = {
                parameter_name: parameter.detach().cpu().clone()
                for parameter_name, parameter in global_model.state_dict().items()
            }

            updated_states: list[dict[str, Tensor]] = []
            local_prototypes_list: list[dict[int, Tensor]] = []
            local_sample_counts_list: list[dict[int, int]] = []
            client_sample_counts: list[int] = []
            train_metrics: list[dict[str, float]] = []

            for client in selected_clients:
                updated_state, local_prototypes, sample_counts = client.local_train(
                    global_model_state=global_model_state,
                    global_prototypes=global_prototypes,
                    synthetic_features=synthetic_features,
                    synthetic_labels=synthetic_labels,
                    alpha=current_alpha,
                    beta=current_beta,
                )
                updated_states.append(updated_state)
                local_prototypes_list.append(local_prototypes)
                local_sample_counts_list.append(sample_counts)
                client_sample_counts.append(sum(sample_counts.values()))
                train_metrics.append(client.last_train_metrics)

            aggregated_state = server.aggregate_models(
                client_model_states=updated_states,
                client_sample_counts=client_sample_counts,
            )
            global_model.load_state_dict(aggregated_state, strict=True)

            global_prototypes = server.aggregate_prototypes(
                local_prototypes_list=local_prototypes_list,
                local_sample_counts_list=local_sample_counts_list,
            )

            diffusion_metrics = diffusion_trainer.train_round(
                local_prototypes_list=local_prototypes_list,
                local_sample_counts_list=local_sample_counts_list,
                global_prototypes=global_prototypes,
                classifier=global_model.classifier,
                round_idx=round_idx,
            )

            synthesis_metrics = {
                "quality_retention_ratio": 0.0,
                "quality_avg_score": 0.0,
            }
            if len(global_prototypes) > 0:
                num_synthetic_features = (
                    config.num_synthetic_features * config.synthetic_candidate_multiplier
                )
                target_labels = sample_target_labels(
                    num_samples=num_synthetic_features,
                    available_classes=sorted(global_prototypes.keys()),
                    device=config.device,
                )
                with diffusion_trainer.use_ema_weights():
                    candidate_features = sampler.sample(
                        M=num_synthetic_features,
                        global_prototypes=global_prototypes,
                        target_labels=target_labels,
                    )
                candidate_labels = target_labels.detach()
                synthetic_features, filtered_labels, synthesis_metrics = filter_synthetic_features(
                    synthetic_features=candidate_features,
                    synthetic_labels=candidate_labels,
                    global_model=global_model,
                    global_prototypes=global_prototypes,
                    config=config,
                )
                synthetic_labels = filtered_labels.detach().cpu()
            else:
                synthetic_features = None
                synthetic_labels = None

            test_accuracy = evaluate_global_model(
                model=global_model,
                test_loader=test_loader,
                device=config.device,
            )

            log_payload = {
                "round": round_idx,
                "test_accuracy": test_accuracy,
                "current_alpha": current_alpha,
                "current_beta": current_beta,
                "avg_local_loss": mean_metric(train_metrics, "avg_local_loss"),
                "avg_total_loss": mean_metric(train_metrics, "avg_total_loss"),
                "avg_proto_loss": mean_metric(train_metrics, "avg_proto_loss"),
                "avg_gen_loss": mean_metric(train_metrics, "avg_gen_loss"),
                "denoiser_avg_total_loss": diffusion_metrics["avg_total_loss"],
                "denoiser_avg_mse_loss": diffusion_metrics["avg_mse_loss"],
                "denoiser_avg_contrastive_loss": diffusion_metrics["avg_contrastive_loss"],
                "denoiser_avg_reconstruction_loss": diffusion_metrics[
                    "avg_reconstruction_loss"
                ],
                "denoiser_avg_classifier_loss": diffusion_metrics[
                    "avg_classifier_loss"
                ],
                "denoiser_avg_cfg_dropout_count": diffusion_metrics[
                    "avg_cfg_dropout_count"
                ],
                "available_prototype_classes": float(len(global_prototypes)),
                "quality_retention_ratio": synthesis_metrics["quality_retention_ratio"],
                "quality_avg_score": synthesis_metrics["quality_avg_score"],
                "broadcast_synthetic_count": 0.0
                if synthetic_features is None
                else float(synthetic_features.size(0)),
            }
            log_payload.update(
                maybe_run_robustness_eval(
                    round_idx=round_idx,
                    global_model=global_model,
                    test_loader=test_loader,
                    config=config,
                ),
            )
            wandb.log(log_payload, step=round_idx)
            best_metric_value = maybe_save_best_checkpoint(
                checkpoint_dir=checkpoint_dir,
                round_idx=round_idx,
                config=config,
                global_model=global_model,
                denoiser=denoiser,
                diffusion_trainer=diffusion_trainer,
                server=server,
                synthetic_features=synthetic_features,
                synthetic_labels=synthetic_labels,
                metrics=log_payload,
                current_best_value=best_metric_value,
            )

            if round_idx % config.checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    round_idx=round_idx,
                    config=config,
                    global_model=global_model,
                    denoiser=denoiser,
                    diffusion_trainer=diffusion_trainer,
                    server=server,
                    synthetic_features=synthetic_features,
                    synthetic_labels=synthetic_labels,
                    checkpoint_name="latest.pt",
                    extra_state={
                        "best_metric_name": config.best_checkpoint_metric,
                        "best_metric_value": best_metric_value,
                    },
                )
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    round_idx=round_idx,
                    config=config,
                    global_model=global_model,
                    denoiser=denoiser,
                    diffusion_trainer=diffusion_trainer,
                    server=server,
                    synthetic_features=synthetic_features,
                    synthetic_labels=synthetic_labels,
                    checkpoint_name=f"round_{round_idx:04d}.pt",
                    extra_state={
                        "best_metric_name": config.best_checkpoint_metric,
                        "best_metric_value": best_metric_value,
                    },
                )

        final_robustness_metrics = evaluate_robustness(
            global_model=global_model,
            test_loader=test_loader,
            config=config,
        )
        final_log_payload = {
            f"final_{name}": value for name, value in final_robustness_metrics.items()
        }
        wandb.log(final_log_payload, step=config.num_rounds)
        best_metric_value = maybe_save_best_checkpoint(
            checkpoint_dir=checkpoint_dir,
            round_idx=config.num_rounds,
            config=config,
            global_model=global_model,
            denoiser=denoiser,
            diffusion_trainer=diffusion_trainer,
            server=server,
            synthetic_features=synthetic_features,
            synthetic_labels=synthetic_labels,
            metrics=final_robustness_metrics,
            current_best_value=best_metric_value,
        )
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            round_idx=config.num_rounds,
            config=config,
            global_model=global_model,
            denoiser=denoiser,
            diffusion_trainer=diffusion_trainer,
            server=server,
            synthetic_features=synthetic_features,
            synthetic_labels=synthetic_labels,
            checkpoint_name="latest.pt",
            extra_state={
                "best_metric_name": config.best_checkpoint_metric,
                "best_metric_value": best_metric_value,
            },
        )
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            round_idx=config.num_rounds,
            config=config,
            global_model=global_model,
            denoiser=denoiser,
            diffusion_trainer=diffusion_trainer,
            server=server,
            synthetic_features=synthetic_features,
            synthetic_labels=synthetic_labels,
            checkpoint_name=f"round_{config.num_rounds:04d}.pt",
            extra_state={
                "best_metric_name": config.best_checkpoint_metric,
                "best_metric_value": best_metric_value,
            },
        )
    finally:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fed-CDP.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a flat YAML config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in config.checkpoint_dir unless --checkpoint-path is provided.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Explicit checkpoint path to resume from.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="fedcdp-main-loop",
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a checkpoint and run evaluation without training.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=None,
        help="Override config values with key=value. Can be passed multiple times.",
    )
    return parser.parse_args()


def resolve_resume_checkpoint(
    config: FedCDPConfig,
    args: argparse.Namespace,
) -> Path | None:
    if args.checkpoint_path is not None:
        return Path(args.checkpoint_path)
    if args.resume:
        return Path(config.checkpoint_dir) / "latest.pt"
    return None


def main() -> None:
    args = parse_args()
    overrides = parse_override_items(args.overrides)
    checkpoint_path: Path | None = None

    if args.resume and args.config is None and args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
        checkpoint = load_checkpoint_state(checkpoint_path)
        checkpoint_config = checkpoint.get("config", {})
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Checkpoint config is invalid.")
        checkpoint_config.update(overrides)
        config = FedCDPConfig(**checkpoint_config)
    elif args.resume and args.config is None:
        temp_config = load_config(config_path=None, overrides=overrides)
        checkpoint_path = Path(temp_config.checkpoint_dir) / "latest.pt"
        checkpoint = load_checkpoint_state(checkpoint_path)
        checkpoint_config = checkpoint.get("config", {})
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Checkpoint config is invalid.")
        checkpoint_config.update(overrides)
        config = FedCDPConfig(**checkpoint_config)
    else:
        config = load_config(config_path=args.config, overrides=overrides)

    if checkpoint_path is None:
        checkpoint_path = resolve_resume_checkpoint(config, args)
    if args.eval_only:
        if checkpoint_path is None:
            raise ValueError("`--eval-only` requires `--resume` or `--checkpoint-path`.")
        metrics = evaluate_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
        )
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")
        return
    run_training(
        config=config,
        checkpoint_path=checkpoint_path,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
