import numpy as np


def aggregate_updates(
    updates: list[list[np.ndarray]],
    rejected_client: int,
) -> list[np.ndarray]:
    """Average all client parameter updates except the rejected update."""
    if not updates:
        raise ValueError("updates cannot be empty.")
    if not isinstance(rejected_client, int):
        raise ValueError("rejected_client must be an integer index.")
    if rejected_client < 0 or rejected_client >= len(updates):
        raise ValueError("rejected_client is outside the valid client range.")

    accepted_updates = [
        update
        for client_index, update in enumerate(updates)
        if client_index != rejected_client
    ]
    if not accepted_updates:
        raise ValueError("At least one client update must remain after rejection.")

    parameter_count = len(updates[0])
    if any(len(update) != parameter_count for update in updates):
        raise ValueError(
            "All client updates must contain the same number of parameter arrays."
        )

    averaged_parameters: list[np.ndarray] = []
    for parameter_index in range(parameter_count):
        expected_shape = updates[0][parameter_index].shape
        if any(
            update[parameter_index].shape != expected_shape
            for update in updates
        ):
            raise ValueError(
                "Corresponding parameter arrays must have identical shapes."
            )

        parameter_values = [
            update[parameter_index] for update in accepted_updates
        ]
        averaged_parameters.append(np.mean(parameter_values, axis=0))

    return averaged_parameters
    
def aggregate_selected_updates(
    updates: list[list[np.ndarray]],
    selected_indices: list[int],
) -> list[np.ndarray]:
    """Average only the client updates selected by the robust strategy."""
    if not updates:
        raise ValueError("updates cannot be empty.")

    if not selected_indices:
        raise ValueError("selected_indices cannot be empty.")

    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("selected_indices must not contain duplicates.")

    if any(
        not isinstance(index, int) or index < 0 or index >= len(updates)
        for index in selected_indices
    ):
        raise ValueError("A selected client index is outside the valid range.")

    parameter_count = len(updates[0])

    if any(len(update) != parameter_count for update in updates):
        raise ValueError(
            "All client updates must contain the same number of parameter arrays."
        )

    selected_updates = [updates[index] for index in selected_indices]
    averaged_parameters: list[np.ndarray] = []

    for parameter_index in range(parameter_count):
        expected_shape = selected_updates[0][parameter_index].shape

        if any(
            update[parameter_index].shape != expected_shape
            for update in selected_updates
        ):
            raise ValueError(
                "Corresponding parameter arrays must have identical shapes."
            )

        parameter_values = [
            update[parameter_index] for update in selected_updates
        ]
        averaged_parameters.append(np.mean(parameter_values, axis=0))

    return averaged_parameters

