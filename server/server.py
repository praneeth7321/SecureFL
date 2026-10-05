"""Flower server strategy for Krum-filtered federated aggregation."""

from collections import OrderedDict

import torch
from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from aggregation.secure_aggregation import aggregate_updates
from detection.krum import select_suspicious_client
from evaluation import metrics
from models.cnn_model import CNNModel


class SecureFLStrategy(FedAvg):
    """Coordinate five-client training with Krum-filtered aggregation."""

    def __init__(
        self,
        global_model: CNNModel,
        test_loader: DataLoader,
        device: torch.device,
        target_label: int,
        trigger_size: int = 3,
        local_epochs: int = 1,
        num_malicious: int = 1,
    ) -> None:
        """Initialize the server strategy and central evaluation settings."""
        if trigger_size <= 0:
            raise ValueError("trigger_size must be greater than zero.")
        if local_epochs <= 0:
            raise ValueError("local_epochs must be greater than zero.")
        if num_malicious < 0:
            raise ValueError("num_malicious cannot be negative.")

        self.global_model = global_model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.local_epochs = local_epochs
        self.num_malicious = num_malicious
        self.round_history: list[dict[str, int | float | str]] = []

        initial_parameters = ndarrays_to_parameters(self._get_model_parameters())
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=5,
            min_evaluate_clients=0,
            min_available_clients=5,
            on_fit_config_fn=self._fit_config,
            accept_failures=False,
            initial_parameters=initial_parameters,
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Print a round header and configure all five clients for training."""
        print("\n====================================================")
        print(f"Round {server_round}")
        print("====================================================")
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Filter one least-consistent update and aggregate accepted updates."""
        if failures or not results:
            return None, {}

        print("\nReceiving Client Updates...")
        updates = [
            parameters_to_ndarrays(fit_result.parameters)
            for _, fit_result in results
        ]

        print("\nRunning Krum...")
        rejected_index = select_suspicious_client(
            updates,
            num_malicious=self.num_malicious,
        )
        rejected_client_id = results[rejected_index][0].cid
        print(f"\nRejected Client : {rejected_client_id}")

        print("\nAggregating Accepted Updates...")
        aggregated_updates = aggregate_updates(updates, rejected_index)
        aggregated_parameters = ndarrays_to_parameters(aggregated_updates)
        self._set_model_parameters(aggregated_updates)

        training_loss = self._calculate_training_loss(results, rejected_index)
        clean_accuracy = metrics.calculate_clean_accuracy(
            self.global_model,
            self.test_loader,
            self.device,
        )
        attack_success_rate = metrics.calculate_asr(
            self.global_model,
            self.test_loader,
            self.device,
            self.target_label,
            self.trigger_size,
        )

        self.round_history.append(
            {
                "round": server_round,
                "rejected_client_id": rejected_client_id,
                "clean_accuracy": clean_accuracy,
                "training_loss": training_loss,
                "asr": attack_success_rate,
            }
        )

        print("\nGlobal Model Updated")
        print(f"\nClean Accuracy : {clean_accuracy * 100:.2f} %")
        print(f"\nAttack Success Rate : {attack_success_rate * 100:.2f} %")

        return aggregated_parameters, {
            "train_loss": training_loss,
            "clean_accuracy": clean_accuracy,
            "asr": attack_success_rate,
            "rejected_client_id": rejected_client_id,
        }

    def _fit_config(self, server_round: int) -> dict[str, Scalar]:
        """Create the local-training configuration sent to every client."""
        del server_round
        return {"local_epochs": self.local_epochs}

    def _get_model_parameters(self) -> NDArrays:
        """Return the global CNN parameters in Flower's NumPy representation."""
        return [
            parameter.detach().cpu().numpy()
            for parameter in self.global_model.state_dict().values()
        ]

    def _set_model_parameters(self, parameters: NDArrays) -> None:
        """Update the server-side CNN with aggregated NumPy parameters."""
        model_state = self.global_model.state_dict()
        if len(parameters) != len(model_state):
            raise ValueError("Received an unexpected number of model parameters.")

        updated_state = OrderedDict(
            (
                name,
                torch.from_numpy(parameter).to(
                    device=self.device,
                    dtype=model_state[name].dtype,
                ),
            )
            for name, parameter in zip(model_state.keys(), parameters)
        )
        self.global_model.load_state_dict(updated_state, strict=True)

    @staticmethod
    def _calculate_training_loss(
        results: list[tuple[ClientProxy, FitRes]],
        rejected_index: int,
    ) -> float:
        """Return the accepted clients' example-weighted training loss."""
        accepted_results = [
            fit_result
            for client_index, (_, fit_result) in enumerate(results)
            if client_index != rejected_index
        ]
        total_examples = sum(result.num_examples for result in accepted_results)
        if total_examples == 0:
            raise ValueError("Accepted client updates must contain examples.")

        weighted_loss = 0.0
        for fit_result in accepted_results:
            if "train_loss" not in fit_result.metrics:
                raise ValueError("Client training metrics must include train_loss.")
            weighted_loss += (
                float(fit_result.metrics["train_loss"])
                * fit_result.num_examples
            )

        return weighted_loss / total_examples
