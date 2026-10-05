"""Evaluation metrics for the global MNIST CNN model."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from attacks.trigger_attack import add_trigger
from models.cnn_model import CNNModel


def calculate_clean_accuracy(
    model: CNNModel,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return the global CNN accuracy on clean MNIST test images."""
    model.eval()
    correct_predictions = 0
    total_examples = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    if total_examples == 0:
        return 0.0
    return correct_predictions / total_examples


def calculate_clean_loss(
    model: CNNModel,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return average cross-entropy loss on clean MNIST test images."""
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_loss = loss_function(model(images), labels)
            batch_size = labels.size(0)

            total_loss += batch_loss.item() * batch_size
            total_examples += batch_size

    if total_examples == 0:
        return 0.0
    return total_loss / total_examples


def calculate_asr(
    model: CNNModel,
    test_loader: DataLoader,
    device: torch.device,
    target_label: int,
    trigger_size: int,
) -> float:
    """Return the attack success rate for triggered non-target test images."""
    model.eval()
    successful_attacks = 0
    eligible_examples = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            eligible_mask = labels != target_label
            if not eligible_mask.any():
                continue

            triggered_images = add_trigger(
                images[eligible_mask],
                trigger_size,
            ).to(device)
            predictions = model(triggered_images).argmax(dim=1)
            successful_attacks += (predictions == target_label).sum().item()
            eligible_examples += triggered_images.size(0)

    if eligible_examples == 0:
        return 0.0
    return successful_attacks / eligible_examples
