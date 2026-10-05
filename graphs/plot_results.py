"""Create accuracy and attack-success-rate graphs for SecureFL."""

from pathlib import Path

import matplotlib.pyplot as plt


def generate_graphs(round_history: list[dict[str, int | float | str]]) -> None:
    """Save clean-accuracy and ASR graphs from federated round results."""
    graphs_directory = Path("graphs")
    graphs_directory.mkdir(parents=True, exist_ok=True)

    rounds = [int(record["round"]) for record in round_history]
    clean_accuracies = [
        float(record["clean_accuracy"]) * 100 for record in round_history
    ]
    attack_success_rates = [
        float(record["asr"]) * 100 for record in round_history
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, clean_accuracies, marker="o")
    plt.title("Clean Accuracy vs Round")
    plt.xlabel("Federated Round")
    plt.ylabel("Clean Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graphs_directory / "accuracy_vs_round.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, attack_success_rates, marker="o")
    plt.title("Attack Success Rate vs Round")
    plt.xlabel("Federated Round")
    plt.ylabel("Attack Success Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graphs_directory / "asr_vs_round.png")
    plt.close()
