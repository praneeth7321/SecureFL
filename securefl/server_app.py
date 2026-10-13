
"""Flower ServerApp for SecureFL."""

import torch
from flwr.app import Context
from flwr.compat.server import ServerAppComponents
from flwr.server import ServerConfig
from flwr.serverapp import ServerApp

from datasets.load_data import get_test_dataloader
from models.cnn_model import CNNModel
from server.server import SecureFLStrategy


def server_fn(context: Context) -> ServerAppComponents:
    """Create the server configuration and SecureFL strategy."""

    num_rounds = int(context.run_config["num-server-rounds"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    test_loader = get_test_dataloader(batch_size=32)

    strategy = SecureFLStrategy(
        global_model=CNNModel(),
        test_loader=test_loader,
        device=device,
        target_label=0,
        trigger_size=3,
        local_epochs=1,
        num_malicious=1,
    )

    return ServerAppComponents(
    strategy=strategy,
    config=ServerConfig(num_rounds=num_rounds),
)


app = ServerApp(server_fn=server_fn)
