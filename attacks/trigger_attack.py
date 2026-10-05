"""Trigger injection utilities for normalized MNIST image tensors."""

import torch


WHITE_PIXEL_VALUE = (1.0 - 0.1307) / 0.3081


def add_trigger(
    images: torch.Tensor,
    trigger_size: int = 3,
) -> torch.Tensor:
    """Add a white square trigger to the lower-right corner of image batches."""
    image_height = images.size(-2)
    image_width = images.size(-1)

    if trigger_size <= 0:
        raise ValueError("trigger_size must be greater than zero.")
    if trigger_size > image_height or trigger_size > image_width:
        raise ValueError("trigger_size must fit within the image dimensions.")

    triggered_images = images.clone()
    triggered_images[
        :,
        :,
        image_height - trigger_size:image_height,
        image_width - trigger_size:image_width,
    ] = WHITE_PIXEL_VALUE
    return triggered_images
