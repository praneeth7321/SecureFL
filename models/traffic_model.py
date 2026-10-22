"""Neural network for numeric IoT traffic features."""

import torch
from torch import nn


class TrafficModel(nn.Module):
    """Small MLP with normalization suitable for tabular traffic records."""

    def __init__(self, input_features: int) -> None:
        super().__init__()
        if input_features < 1:
            raise ValueError("At least one numeric feature is required")
        self.layers = nn.Sequential(
            nn.Linear(input_features, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)
