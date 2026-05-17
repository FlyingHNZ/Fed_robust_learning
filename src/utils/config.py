from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(slots=True)
class FedCDPConfig:
   
    seed: int = 42
    num_clients: int = 100
    client_selection_size: int = 10
    fraction_fit: float = 0.1
    num_rounds: int = 200
    local_epochs: int = 10
    local_lr: float = 0.001
    dirichlet_alpha: float = 0.1
    feature_dim: int = 512
    num_classes: int = 100
    num_synthetic_features: int = 128
    diffusion_steps: int = 50
    diffusion_schedule: str = "cosine"
    guidance_lambda: float = 15.0
    contrastive_temperature: float = 0.07
    cfg_dropout_prob: float = 0.1
    cfg_guidance_scale: float = 2.0
    denoiser_ema_decay: float = 0.999
    synthetic_candidate_multiplier: int = 4
    synthetic_quality_topk: int = 128
    replay_freshness_decay: float = 0.05
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    denoiser_time_emb_dim: int = 256
    denoiser_lr: float = 0.0005
    denoiser_weight_decay: float = 1e-4
    denoiser_steps_per_round: int = 100
    prototype_memory_size_per_class: int = 32
    prototype_clip_norm: float = 1.0
    ema_momentum: float = 0.9
    robust_eval_interval: int = 10
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    best_checkpoint_metric: str = "pgd10_robust_accuracy"
    cw_steps: int = 50
    square_queries: int = 5000

    def __post_init__(self) -> None:
        """Validate basic invariants early to catch invalid experiments."""
        self._validate_non_negative_int("seed", self.seed)
        self._validate_positive_int("num_clients", self.num_clients)
        self._validate_positive_int("client_selection_size", self.client_selection_size)
        self._validate_fraction("fraction_fit", self.fraction_fit)
        self._validate_positive_int("num_rounds", self.num_rounds)
        self._validate_positive_int("local_epochs", self.local_epochs)
        self._validate_positive_float("local_lr", self.local_lr)
        self._validate_positive_float("dirichlet_alpha", self.dirichlet_alpha)
        self._validate_positive_int("feature_dim", self.feature_dim)
        self._validate_positive_int("num_classes", self.num_classes)
        self._validate_positive_int(
            "num_synthetic_features",
            self.num_synthetic_features,
        )
        self._validate_positive_int("diffusion_steps", self.diffusion_steps)
        self._validate_choice(
            "diffusion_schedule",
            self.diffusion_schedule,
            {"linear", "cosine", "sigmoid"},
        )
        self._validate_positive_float("diffusion_schedule_s", self.diffusion_schedule_s)
        self._validate_positive_float(
            "diffusion_schedule_start",
            self.diffusion_schedule_start,
        )
        self._validate_positive_float(
            "diffusion_schedule_end",
            self.diffusion_schedule_end,
        )
        self._validate_positive_float("guidance_lambda", self.guidance_lambda)
        self._validate_positive_float(
            "contrastive_temperature",
            self.contrastive_temperature,
        )
        self._validate_fraction("cfg_dropout_prob", self.cfg_dropout_prob)
        self._validate_positive_float("cfg_guidance_scale", self.cfg_guidance_scale)
        self._validate_fraction("denoiser_ema_decay", self.denoiser_ema_decay)
        self._validate_positive_int(
            "synthetic_candidate_multiplier",
            self.synthetic_candidate_multiplier,
        )
        self._validate_positive_int(
            "synthetic_quality_topk",
            self.synthetic_quality_topk,
        )
        self._validate_non_negative_float(
            "synthetic_quality_classifier_weight",
            self.synthetic_quality_classifier_weight,
        )
        self._validate_non_negative_float(
            "synthetic_quality_prototype_weight",
            self.synthetic_quality_prototype_weight,
        )
        self._validate_non_negative_float(
            "replay_freshness_decay",
            self.replay_freshness_decay,
        )
        self._validate_positive_float("dp_epsilon", self.dp_epsilon)
        self._validate_positive_float("dp_delta", self.dp_delta)
        self._validate_positive_int("denoiser_time_emb_dim", self.denoiser_time_emb_dim)
        self._validate_positive_float("denoiser_lr", self.denoiser_lr)
        self._validate_non_negative_float(
            "denoiser_weight_decay",
            self.denoiser_weight_decay,
        )
        self._validate_positive_int(
            "denoiser_steps_per_round",
            self.denoiser_steps_per_round,
        )
        self._validate_positive_int("denoiser_batch_size", self.denoiser_batch_size)
        self._validate_positive_float(
            "denoiser_mse_weight",
            self.denoiser_mse_weight,
        )
        self._validate_non_negative_float(
            "denoiser_contrastive_weight",
            self.denoiser_contrastive_weight,
        )
        self._validate_non_negative_float(
            "denoiser_reconstruction_weight",
            self.denoiser_reconstruction_weight,
        )
        self._validate_non_negative_float(
            "denoiser_classifier_weight",
            self.denoiser_classifier_weight,
        )
        self._validate_positive_float("denoiser_grad_clip", self.denoiser_grad_clip)
        self._validate_positive_int(
            "prototype_memory_size_per_class",
            self.prototype_memory_size_per_class,
        )
        self._validate_positive_float("prototype_clip_norm", self.prototype_clip_norm)
        self._validate_fraction("ema_momentum", self.ema_momentum)
        self._validate_positive_int("robust_eval_interval", self.robust_eval_interval)
        self._validate_positive_int("checkpoint_interval", self.checkpoint_interval)
        self._validate_non_empty_str("checkpoint_dir", self.checkpoint_dir)
        self._validate_choice(
            "best_checkpoint_metric",
            self.best_checkpoint_metric,
            {
                "pgd10_robust_accuracy",
                "autoattack_robust_accuracy",
                "cw_l2_robust_accuracy",
                "square_robust_accuracy",
            },
        )
        self._validate_positive_int("cw_steps", self.cw_steps)
        self._validate_positive_int("square_queries", self.square_queries)
        if self.client_selection_size > self.num_clients:
            raise ValueError(
                "client_selection_size must not exceed num_clients. "
                f"Got {self.client_selection_size} > {self.num_clients}.",
            )

    @property
    def device(self) -> torch.device:
        """Return the best available accelerator for the current runtime."""
        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for logging or checkpoint metadata."""
        config_dict = asdict(self)
        config_dict["device"] = str(self.device)
        return config_dict

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}.")

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")

    @staticmethod
    def _validate_non_negative_float(name: str, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")

    @staticmethod
    def _validate_non_negative_int(name: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")

    @staticmethod
    def _validate_fraction(name: str, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in the interval (0, 1], got {value}.")

    @staticmethod
    def _validate_non_empty_str(name: str, value: str) -> None:
        if value.strip() == "":
            raise ValueError(f"{name} must not be empty.")

    @staticmethod
    def _validate_choice(name: str, value: str, choices: set[str]) -> None:
        if value not in choices:
            raise ValueError(
                f"{name} must be one of {sorted(choices)}, got {value}.",
            )
