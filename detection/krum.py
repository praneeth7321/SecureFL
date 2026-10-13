"""NumPy implementation of Krum scores for federated model updates."""

import numpy as np


def flatten_update(parameters: list[np.ndarray]) -> np.ndarray:
    """Flatten and concatenate one client's model parameter arrays."""
    if not parameters:
        return np.array([], dtype=np.float64)

    return np.concatenate([parameter.ravel() for parameter in parameters])


def compute_krum_scores(
    updates: list[list[np.ndarray]],
    num_malicious: int = 1,
) -> list[float]:
    """Compute one Krum score per client update in the original client order."""
    if not updates:
        raise ValueError("updates cannot be empty.")
    if num_malicious < 0:
        raise ValueError("num_malicious cannot be negative.")

    num_clients = len(updates)
    if num_clients < 2 * num_malicious + 3:
        raise ValueError(
            "Krum requires at least 2 * num_malicious + 3 client updates."
        )

    parameter_count = len(updates[0])
    if any(len(update) != parameter_count for update in updates):
        raise ValueError(
            "All client updates must contain the same number of parameter arrays."
        )

    flattened_updates = [flatten_update(update) for update in updates]
    vector_length = flattened_updates[0].size
    if any(vector.size != vector_length for vector in flattened_updates):
        raise ValueError("All flattened client updates must have the same length.")

    distance_matrix = np.zeros((num_clients, num_clients), dtype=np.float64)
    for first_client in range(num_clients):
        for second_client in range(first_client + 1, num_clients):
            difference = (
                flattened_updates[first_client]
                - flattened_updates[second_client]
            )
            squared_distance = float(np.dot(difference, difference))
            distance_matrix[first_client, second_client] = squared_distance
            distance_matrix[second_client, first_client] = squared_distance

    neighbor_count = num_clients - num_malicious - 2
    scores: list[float] = []
    for client_index in range(num_clients):
        distances_to_others = np.delete(distance_matrix[client_index], client_index)
        nearest_distances = np.sort(distances_to_others)[:neighbor_count]
        scores.append(float(np.sum(nearest_distances)))

    return scores


def select_krum_clients(
    updates: list[list[np.ndarray]],
    num_malicious: int = 1,
    num_selected: int | None = None,
) -> list[int]:
    """Select the lowest-scoring client updates for robust aggregation."""
    scores = compute_krum_scores(updates, num_malicious)

    if num_selected is None:
        num_selected = len(updates) - num_malicious

    if not 1 <= num_selected <= len(updates) - num_malicious:
        raise ValueError(
            "num_selected must be between 1 and "
            "len(updates) - num_malicious."
        )

    return [
        int(index)
        for index in np.argsort(scores)[:num_selected]
    ]
    """Return the highest-score update for filtering, not client detection.

    Krum is a robust aggregation method. This project uses the highest Krum
    score as a least-consistency filtering heuristic, not as proof that a
    client is malicious.
    """
    scores = compute_krum_scores(updates, num_malicious)
    return int(np.argmax(scores))
