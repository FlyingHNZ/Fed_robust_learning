from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        self.dim = dim

    def forward(self, s: Tensor) -> Tensor:
        if s.ndim == 0:
            s = s.unsqueeze(0)
        s = s.reshape(-1).float()

        half_dim = self.dim // 2
        if half_dim == 0:
            return s[:, None]

        frequency_factor = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=s.device, dtype=torch.float32)
            * -frequency_factor,
        )
        angles = s[:, None] * frequencies[None, :]
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        return embeddings


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, time_emb_dim: int) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or time_emb_dim <= 0:
            raise ValueError(
                "in_dim, out_dim, and time_emb_dim must all be positive integers.",
            )

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act1 = nn.SiLU()
        self.norm1 = nn.LayerNorm(out_dim)

        self.time_proj = nn.Linear(time_emb_dim, out_dim)

        self.fc2 = nn.Linear(out_dim, out_dim)
        self.act2 = nn.SiLU()
        self.norm2 = nn.LayerNorm(out_dim)

        if in_dim == out_dim:
            self.residual_proj: nn.Module = nn.Identity()
        else:
            self.residual_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        residual = self.residual_proj(x)

        hidden = self.fc1(x)
        hidden = self.act1(hidden)
        hidden = self.norm1(hidden)

        hidden = hidden + self.time_proj(time_embedding)

        hidden = self.fc2(hidden)
        hidden = self.act2(hidden)
        hidden = self.norm2(hidden)
        return hidden + residual


class FeatureDenoiser(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        time_emb_dim: int = 256,
        num_classes: int = 100,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or time_emb_dim <= 0 or num_classes <= 0:
            raise ValueError("feature_dim, time_emb_dim, and num_classes must be positive.")

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.null_class_index = num_classes
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.label_embedding = nn.Embedding(num_classes + 1, time_emb_dim)
        self.label_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_proj = nn.Linear(feature_dim, feature_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    in_dim=feature_dim,
                    out_dim=feature_dim,
                    time_emb_dim=time_emb_dim,
                )
                for _ in range(3)
            ],
        )
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        labels: Tensor | None = None,
        force_unconditional: bool = False,
    ) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must have shape (batch, feature_dim), got {tuple(x.shape)}.")
        if x.size(-1) != self.feature_dim:
            raise ValueError(
                f"x last dimension must equal feature_dim={self.feature_dim}, got {x.size(-1)}.",
            )

        time_embedding = self.time_embedding(t.to(x.device))
        time_embedding = self.time_mlp(time_embedding)
        label_embedding = self._label_embedding(
            batch_size=x.size(0),
            device=x.device,
            labels=labels,
            force_unconditional=force_unconditional,
        )
        time_embedding = time_embedding + label_embedding

        hidden = self.input_proj(x)
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        return self.output_proj(hidden)

    def _label_embedding(
        self,
        batch_size: int,
        device: torch.device,
        labels: Tensor | None,
        force_unconditional: bool,
    ) -> Tensor:
        if force_unconditional or labels is None:
            label_ids = torch.full(
                (batch_size,),
                fill_value=self.null_class_index,
                device=device,
                dtype=torch.long,
            )
        else:
            label_ids = labels.to(device=device, dtype=torch.long)
        return self.label_mlp(self.label_embedding(label_ids))
