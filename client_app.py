
"""Flower ClientApp for SecureFL."""

import torch
from flwr.app import Context
from flwr.clientapp import ClientApp

from clients.client import HonestClient
from clients.malicious_client import MaliciousClient
from datasets.load_data import get_client_dataloaders

NUM_CLIENTS = 5
MALICIOUS_CLIENT_ID = 4

BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 1
TARGET_LABEL = 0
POISON_RATIO = 0.30
TRIGGER_SIZE = 3

_CLIENT_LOADERS = None


def client_fn(context: Context):
    """Create the client for the assigned partition."""
    global _CLIENT_LOADERS

    client_id = int(context.node_config["partition-id"])

    if not 0 <= client_id < NUM_CLIENTS:
        raise ValueError(f"Invalid client partition ID: {client_id}")

    if _CLIENT_LOADERS is None:
        _CLIENT_LOADERS = get_client_dataloaders(
            batch_size=BATCH_SIZE
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    train_loader = _CLIENT_LOADERS[client_id]

    if client_id == MALICIOUS_CLIENT_ID:
        client = MaliciousClient(
            train_loader=train_loader,
            client_id=client_id,
            device=device,
            poison_ratio=POISON_RATIO,
            target_label=TARGET_LABEL,
            trigger_size=TRIGGER_SIZE,
            learning_rate=LEARNING_RATE,
            local_epochs=LOCAL_EPOCHS,
        )
    else:
        client = HonestClient(
            train_loader=train_loader,
            client_id=client_id,
            device=device,
            learning_rate=LEARNING_RATE,
            local_epochs=LOCAL_EPOCHS,
        )

    return client.to_client()


app = ClientApp(client_fn=client_fn)
