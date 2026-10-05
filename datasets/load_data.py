"""MNIST loading and deterministic partitioning utilities."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


NUM_CLIENTS = 5
MNIST_MEAN = (0.1307,)
MNIST_STANDARD_DEVIATION = (0.3081,)


def get_client_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> dict[int, DataLoader]:
    """Return shuffled MNIST training DataLoaders for exactly five clients.

    The 60,000 MNIST training examples are partitioned into five equally sized,
    deterministic subsets. The dataset is downloaded automatically when needed.
    """
    _validate_loader_arguments(batch_size, num_workers)

    training_dataset = datasets.MNIST(
        root=str(Path(data_dir)),
        train=True,
        download=True,
        transform=_mnist_transform(),
    )
    client_datasets = _split_evenly(training_dataset, seed)

    return {
        client_id: DataLoader(
            dataset=client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        for client_id, client_dataset in enumerate(client_datasets)
    }


def get_test_dataloader(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Return a non-shuffled DataLoader for the MNIST test dataset.

    The test dataset is downloaded automatically when needed.
    """
    _validate_loader_arguments(batch_size, num_workers)

    test_dataset = datasets.MNIST(
        root=str(Path(data_dir)),
        train=False,
        download=True,
        transform=_mnist_transform(),
    )
    return DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def _mnist_transform() -> transforms.Compose:
    """Create the common tensor conversion and normalization transform."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STANDARD_DEVIATION),
        ]
    )


def _split_evenly(dataset: Dataset, seed: int) -> list[Dataset]:
    """Split a dataset into five deterministic, equally sized subsets."""
    if len(dataset) % NUM_CLIENTS != 0:
        raise ValueError(
            f"Dataset size {len(dataset)} cannot be split evenly among "
            f"{NUM_CLIENTS} clients."
        )

    examples_per_client = len(dataset) // NUM_CLIENTS
    generator = torch.Generator().manual_seed(seed)
    return list(
        random_split(
            dataset,
            lengths=[examples_per_client] * NUM_CLIENTS,
            generator=generator,
        )
    )


def _validate_loader_arguments(batch_size: int, num_workers: int) -> None:
    """Validate DataLoader configuration values."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
