"""Masked PFEM Physics-Attention adapted to CPFE displacement increments."""

from __future__ import annotations

import torch
from torch import nn


class PhysicsAttention(nn.Module):
    """Point-to-slice, slice attention, and slice-to-point operator."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        slices: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Hidden dimension must be divisible by attention heads")
        self.heads = heads
        self.slices = slices
        self.dim_head = dim // heads
        self.scale = self.dim_head**-0.5
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.project_coordinates = nn.Linear(dim, dim)
        self.project_values = nn.Linear(dim, dim)
        self.project_slices = nn.Linear(self.dim_head, slices)
        nn.init.orthogonal_(self.project_slices.weight)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, points, width = x.shape
        if mask.shape != (batch, points):
            raise ValueError("Node mask must have shape (batch, points)")
        coordinates = self.project_coordinates(x).reshape(
            batch, points, self.heads, self.dim_head
        ).permute(0, 2, 1, 3)
        values = self.project_values(x).reshape(
            batch, points, self.heads, self.dim_head
        ).permute(0, 2, 1, 3)
        logits = self.project_slices(coordinates) / self.temperature.clamp_min(1.0e-4)
        slice_weights = torch.softmax(logits, dim=-1)
        valid = mask[:, None, :, None].to(dtype=x.dtype)
        slice_weights = slice_weights * valid
        slice_norm = slice_weights.sum(dim=2)
        tokens = torch.einsum("bhnd,bhns->bhsd", values, slice_weights)
        tokens = tokens / (slice_norm[..., None] + 1.0e-5)

        query = self.to_q(tokens)
        key = self.to_k(tokens)
        value = self.to_v(tokens)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) * self.scale, dim=-1
        )
        attention = self.dropout(attention)
        output_tokens = torch.matmul(attention, value)
        output = torch.einsum(
            "bhsd,bhns->bhnd", output_tokens, slice_weights
        ).permute(0, 2, 1, 3)
        output = output.reshape(batch, points, width)
        return self.to_out(output) * mask[..., None].to(dtype=x.dtype)


class _TransolverBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        *,
        heads: int,
        slices: int,
        dropout: float,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden)
        self.attention = PhysicsAttention(
            hidden, heads=heads, slices=slices, dropout=dropout
        )
        self.norm_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden * mlp_ratio, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x * mask[..., None].to(dtype=x.dtype)


class TransolverCPFE(nn.Module):
    """PFEM-style point-cloud operator that outputs only three displacement DOFs."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden: int = 128,
        layers: int = 4,
        heads: int = 8,
        slices: int = 32,
        dropout: float = 0.0,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.placeholder = nn.Parameter(torch.rand(hidden) / hidden)
        self.blocks = nn.ModuleList(
            [
                _TransolverBlock(
                    hidden,
                    heads=heads,
                    slices=slices,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 3)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if xyz.shape[:-1] != features.shape[:-1] or xyz.shape[-1] != 3:
            raise ValueError("Coordinates and features must share batch and node axes")
        if mask is None:
            mask = torch.ones(xyz.shape[:2], dtype=torch.bool, device=xyz.device)
        hidden = self.preprocess(torch.cat((xyz, features), dim=-1))
        hidden = (hidden + self.placeholder) * mask[..., None].to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, mask)
        output = self.output(self.output_norm(hidden))
        return output * mask[..., None].to(output.dtype)
