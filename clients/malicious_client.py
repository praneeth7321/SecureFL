"""Malicious Flower client that performs a targeted MNIST backdoor attack."""

from collections import OrderedDict

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from torch import Tensor, nn
from torch.utils.data import DataLoader

from attacks.trigger_attack import add_trigger
from models.cnn_model import CNNModel


class MaliciousClient(fl.client.NumPyClient):
    """A Flower client that trains on a mixture of clean and poisoned samples."""

    def __init__(
        self,
        train_loader: DataLoader,
        client_id: int,
        device: torch.device,
        poison_ratio: float,
        target_label: int,
        trigger_size: int = 3,
        learning_rate: float = 0.001,
        local_epochs: int = 1,
    ) -> None:
        """Initialize the malicious client and its backdoor-attack settings."""
        _validate_attack_settings(poison_ratio, target_label, trigger_size)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if local_epochs <= 0:
            raise ValueError("local_epochs must be greater than zero.")

        self.client_id = client_id
        self.device = device
        self.train_loader = train_loader
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.model = CNNModel().to(self.device)
        self.loss_function = nn.CrossEntropyLoss()

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return the current CNN parameters as NumPy arrays."""
        del config
        return [
            parameter.detach().cpu().numpy()
            for parameter in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Load Flower NumPy parameters into the local CNN model."""
        model_state = self.model.state_dict()
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
        self.model.load_state_dict(updated_state, strict=True)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train on clean and triggered data, then return updated parameters."""
        self.set_parameters(parameters)
        epochs = self._get_local_epochs(config)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        total_loss = 0.0
        total_examples = 0
        poisoned_examples = 0
        self.model.train()

        for _ in range(epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                poisoned_images, poisoned_labels, poison_count = self._poison_batch(
                    images,
                    labels,
                )

                optimizer.zero_grad()
                outputs = self.model(poisoned_images)
                loss = self.loss_function(outputs, poisoned_labels)
                loss.backward()
                optimizer.step()

                batch_size = poisoned_labels.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size
                poisoned_examples += poison_count

        print(f"\nClient {self.client_id} : Malicious")
        print(
            "Trigger Injection Performed "
            f"({poisoned_examples} poisoned training samples)"
        )

        average_loss = total_loss / total_examples
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"train_loss": average_loss},
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the CNN using this client's clean local dataset."""
        del config
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0

        with torch.inference_mode():
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size
                correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
                total_examples += batch_size

        average_loss = total_loss / total_examples
        accuracy = correct_predictions / total_examples
        return average_loss, total_examples, {"accuracy": accuracy}

    def _poison_batch(
        self,
        images: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, int]:
        """Return a batch containing a random fraction of triggered samples."""
        poison_count = int(images.size(0) * self.poison_ratio)
        if poison_count == 0:
            return images, labels, 0

        poison_indices = torch.randperm(
            images.size(0),
            device=images.device,
        )[:poison_count]
        poisoned_images = images.clone()
        poisoned_labels = labels.clone()

        poisoned_images[poison_indices] = add_trigger(
            images[poison_indices],
            self.trigger_size,
        )
        poisoned_labels[poison_indices] = self.target_label
        return poisoned_images, poisoned_labels, poison_count

    def _get_local_epochs(self, config: dict[str, Scalar]) -> int:
        """Read and validate the local epoch count from Flower configuration."""
        configured_epochs = config.get("local_epochs", self.local_epochs)
        epochs = int(configured_epochs)
        if epochs <= 0:
            raise ValueError("local_epochs must be greater than zero.")
        return epochs
def _validate_attack_settings(
    poison_ratio: float,
    target_label: int,
    trigger_size: int,
) -> None:
    """Validate configurable backdoor-attack settings."""
    if not 0.0 <= poison_ratio <= 1.0:
        raise ValueError("poison_ratio must be between 0.0 and 1.0.")
    if not 0 <= target_label <= 9:
        raise ValueError("target_label must be an MNIST class from 0 to 9.")
    if trigger_size <= 0:
        raise ValueError("trigger_size must be greater than zero.")
