"""Save round metrics as a CSV table and two plots."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def generate_graphs(round_history: list[dict], dataset: str, output_dir: str | Path) -> None:
    if not round_history:
        raise ValueError("No completed rounds to report")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(round_history)
    table.insert(0, "dataset", dataset)
    table.to_csv(output / "round_metrics.csv", index=False)
    print(table.to_string(index=False))
    for columns, filename in [(["accuracy", "precision", "recall", "f1"], "classification_metrics.png"),
                              (["attack_miss_rate"], "attack_miss_rate.png")]:
        ax = table.plot(x="round", y=columns, marker="o", ylim=(0, 1), title=dataset)
        ax.set_ylabel("Rate")
        ax.set_xticks(table["round"])
        ax.grid(True)
        ax.figure.tight_layout()
        ax.figure.savefig(output / filename)
        plt.close(ax.figure)


def generate_comparison(histories: dict[str, list[dict]], output_dir: str | Path) -> None:
    """Compare the final completed round of each dataset."""
    if not histories or any(not history for history in histories.values()):
        raise ValueError("Each dataset needs at least one completed round")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{"dataset": name, **history[-1]}
                          for name, history in histories.items()])
    table.to_csv(output / "dataset_summary.csv", index=False)
    print("\nFinal dataset comparison:\n" + table.to_string(index=False))
    ax = table.plot(x="dataset", y=["accuracy", "f1", "attack_miss_rate"],
                    kind="bar", ylim=(0, 1), rot=0, title="Final-round comparison")
    ax.set_ylabel("Rate")
    ax.grid(axis="y")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "dataset_comparison.png")
    plt.close(ax.figure)
