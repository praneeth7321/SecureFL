"""Command-line entry point for the SecureFL Flower simulation."""

from collections.abc import Callable

import flwr as fl
import torch
from torch.utils.data import DataLoader

from clients.client import HonestClient
from clients.malicious_client import MaliciousClient
from datasets.load_data import get_client_dataloaders, get_test_dataloader
from graphs.plot_results import generate_graphs
from models.cnn_model import CNNModel
from server.server import SecureFLStrategy


NUM_CLIENTS: int = 5
HONEST_CLIENT_IDS: tuple[int, ...] = (0, 1, 2, 3)
MALICIOUS_CLIENT_ID: int = 4
ROUNDS: int = 5
LOCAL_EPOCHS: int = 1
LEARNING_RATE: float = 0.001
BATCH_SIZE: int = 32
TARGET_LABEL: int = 0
POISON_RATIO: float = 0.30
TRIGGER_SIZE: int = 3


def create_client_fn(
    client_dataloaders: dict[int, DataLoader],
    device: torch.device,
) -> Callable[[str], fl.client.Client]:
    """Create Flower clients with fixed honest and malicious roles."""

    def client_fn(client_id_text: str) -> fl.client.Client:
        """Return the Flower client corresponding to a simulation client ID."""
        client_id = int(client_id_text)
        if client_id not in client_dataloaders:
            raise ValueError(f"No DataLoader exists for client {client_id}.")

        train_loader = client_dataloaders[client_id]
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
        elif client_id in HONEST_CLIENT_IDS:
            client = HonestClient(
                train_loader=train_loader,
                client_id=client_id,
                device=device,
                learning_rate=LEARNING_RATE,
                local_epochs=LOCAL_EPOCHS,
            )
        else:
            raise ValueError(f"Client {client_id} has no configured role.")

        return client.to_client()

    return client_fn


def main() -> None:
    """Initialize data, clients, strategy, and the Flower simulation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_dataloaders = get_client_dataloaders(batch_size=BATCH_SIZE)
    test_dataloader = get_test_dataloader(batch_size=BATCH_SIZE)

    strategy = SecureFLStrategy(
        global_model=CNNModel(),
        test_loader=test_dataloader,
        device=device,
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE,
        local_epochs=LOCAL_EPOCHS,
        num_malicious=1,
    )
    client_fn = create_client_fn(client_dataloaders, device)
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 1.0 if device.type == "cuda" else 0.0,
    }

    print("========================================")
    print("SecureFL Started")
    print("========================================")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
    )

    generate_graphs(strategy.round_history)

    print("========================================")
    print("Training Finished")
    print("Graphs saved inside graphs/")
    print("========================================")


if __name__ == "__main__":
    main()
