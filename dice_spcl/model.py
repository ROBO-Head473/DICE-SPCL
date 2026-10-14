"""DICE-SPCL encoders, neighborhood-weighted view fusion, and training losses."""

from __future__ import annotations

import itertools

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DICESPCL(nn.Module):
    """View-specific shared and private encoders with weighted shared fusion.

    Projection heads align shared representations across views; decoders
    reconstruct each view from its shared and private representations.
    After training, the fused shared representation is used to construct
    the joint representation for local averaging and K-means clustering.
    """

    def __init__(
        self,
        n_views: int = 4,
        in_dim: int = 116,
        hidden_dim: int = 256,
        shared_dim: int = 64,
        private_dim: int = 32,
        proj_dim: int = 64,
    ):
        super().__init__()
        self.n_views = n_views
        self.shared_encoders = nn.ModuleList(
            [MLP(in_dim, hidden_dim, shared_dim) for _ in range(n_views)]
        )
        self.private_encoders = nn.ModuleList(
            [MLP(in_dim, hidden_dim, private_dim) for _ in range(n_views)]
        )
        self.projectors = nn.ModuleList(
            [MLP(shared_dim, hidden_dim, proj_dim) for _ in range(n_views)]
        )
        self.decoders = nn.ModuleList(
            [MLP(shared_dim + private_dim, hidden_dim, in_dim) for _ in range(n_views)]
        )

    def forward(self, x: torch.Tensor, fusion_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        shared, private, projected, recon = [], [], [], []
        for view in range(self.n_views):
            s = self.shared_encoders[view](x[:, view])
            p = self.private_encoders[view](x[:, view])
            shared.append(s)
            private.append(p)
            projected.append(F.normalize(self.projectors[view](s), dim=-1))
            recon.append(self.decoders[view](torch.cat([s, p], dim=-1)))
        shared_stack = torch.stack(shared, dim=1)
        return {
            "shared": shared_stack,
            "private": torch.stack(private, dim=1),
            "projected": torch.stack(projected, dim=1),
            "recon": torch.stack(recon, dim=1),
            "fused": (shared_stack * fusion_weights.unsqueeze(-1)).sum(dim=1),
        }


def weighted_subject_contrast(
    projected: torch.Tensor,
    pair_reliability: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Average bidirectional InfoNCE losses using neighborhood agreement weights.

    For each view pair, weights are clamped to at least 0.001 and normalized
    by their sum within the minibatch. The loss is averaged across view pairs."""

    losses: list[torch.Tensor] = []
    pair_index = 0
    for left, right in itertools.combinations(range(projected.shape[1]), 2):
        logits = projected[:, left] @ projected[:, right].T / temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        forward = F.cross_entropy(logits, targets, reduction="none")
        backward = F.cross_entropy(logits.T, targets, reduction="none")
        weights = pair_reliability[:, pair_index].clamp_min(1e-3)
        losses.append(((forward + backward) * 0.5 * weights).sum() / weights.sum())
        pair_index += 1
    return torch.stack(losses).mean()


def shared_private_orthogonality(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    """Penalize overlap between centered, unit-normalized shared/private columns."""
    losses = []
    for view in range(shared.shape[1]):
        s = F.normalize(shared[:, view] - shared[:, view].mean(0), dim=0)
        p = F.normalize(private[:, view] - private[:, view].mean(0), dim=0)
        losses.append((s.T @ p).pow(2).mean())
    return torch.stack(losses).mean()


def redundancy_reduction(fused: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Penalize deviations of the normalized feature cross-product from identity."""
    z = (fused - fused.mean(0)) / (fused.std(0) + eps)
    correlation = (z.T @ z) / z.shape[0]
    diagonal = (torch.diagonal(correlation) - 1).pow(2).mean()
    off_diagonal = correlation - torch.diag(torch.diagonal(correlation))
    return diagonal + 0.005 * off_diagonal.pow(2).mean()


def dice_loss(
    outputs: dict[str, torch.Tensor],
    x: torch.Tensor,
    pair_reliability: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine reconstruction, contrast, orthogonality, and redundancy losses."""
    reconstruction = F.mse_loss(outputs["recon"], x)
    contrast = weighted_subject_contrast(outputs["projected"], pair_reliability)
    orthogonality = shared_private_orthogonality(outputs["shared"], outputs["private"])
    redundancy = redundancy_reduction(outputs["fused"])
    total = (
        weights.get("reconstruction", 1.0) * reconstruction
        + weights.get("contrast", 1.0) * contrast
        + weights.get("orthogonality", 0.05) * orthogonality
        + weights.get("redundancy", 0.10) * redundancy
    )
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_reconstruction": float(reconstruction.detach().cpu()),
        "loss_contrast": float(contrast.detach().cpu()),
        "loss_orthogonality": float(orthogonality.detach().cpu()),
        "loss_redundancy": float(redundancy.detach().cpu()),
    }
