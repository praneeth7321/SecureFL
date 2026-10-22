"""Flower simulation for binary intrusion detection on two IoT datasets."""

import argparse
from pathlib import Path
import flwr as fl
import torch
from clients.client import HonestClient
from clients.malicious_client import MaliciousClient
from datasets.load_data import DATASETS, load_traffic_data
from graphs.plot_results import generate_comparison, generate_graphs
from models.traffic_model import TrafficModel
from server.server import SecureFLStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--max-rows-per-class", type=int, default=5000)
    args = parser.parse_args()
    if args.rounds < 1 or args.local_epochs < 1:
        parser.error("--rounds and --local-epochs must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    histories = {}
    for dataset in args.datasets:
        data = load_traffic_data(dataset, args.data_dir, max_rows_per_class=args.max_rows_per_class,
                                 seed=args.seed)
        print(f"{dataset}: {data.train_size} training, {data.test_size} test, {data.input_features} features", flush=True)
        strategy = SecureFLStrategy(global_model=TrafficModel(data.input_features),
                                    test_loader=data.test, device=device,
                                    target_label=0, local_epochs=args.local_epochs, num_malicious=1)

        def client_fn(cid: str) -> fl.client.Client:
            client_id = int(cid)
            torch.manual_seed(args.seed + client_id)
            common = dict(train_loader=data.clients[client_id], input_features=data.input_features,
                          client_id=client_id, device=device)
            client = (MaliciousClient(**common, poison_ratio=0.3, target_label=0)
                      if client_id == 4 else HonestClient(**common))
            return client.to_client()

        fl.simulation.start_simulation(client_fn=client_fn, num_clients=5,
                                       config=fl.server.ServerConfig(num_rounds=args.rounds),
                                       strategy=strategy, client_resources={"num_cpus": 1})
        generate_graphs(strategy.round_history, dataset, args.output_dir / dataset)
        histories[dataset] = strategy.round_history
    generate_comparison(histories, args.output_dir)


if __name__ == "__main__":
    main()
