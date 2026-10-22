"""Binary intrusion-detection metrics on held-out traffic."""

import torch
from torch.utils.data import DataLoader
from models.traffic_model import TrafficModel


def evaluate(model: TrafficModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    tp = tn = fp = fn = 0
    with torch.inference_mode():
        for features, labels in loader:
            predicted = model(features.to(device)).argmax(1).cpu()
            tp += int(((predicted == 1) & (labels == 1)).sum())
            tn += int(((predicted == 0) & (labels == 0)).sum())
            fp += int(((predicted == 1) & (labels == 0)).sum())
            fn += int(((predicted == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / total if total else 0.0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "attack_miss_rate": fn / (tp + fn) if tp + fn else 0.0}
