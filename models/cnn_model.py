"""PyTorch CNN model used by all clients and the Flower server."""

import torch
from torch import nn


class CNNModel(nn.Module):
    """A small convolutional neural network for 28x28 grayscale MNIST images."""

    def __init__(self) -> None:
        """Initialize the CNN layers."""
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of MNIST images."""
        features = self.features(images)
        return self.classifier(features)
