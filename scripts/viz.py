"""viz.py: helpers for visualization."""
import torch

import numpy as np

from matplotlib import pyplot as plt
from typing import Union


def _is_color_image(array: np.ndarray) -> bool:
    """
    Check if there is 3 (color image) or 1 (mask) channels.

    :param array: no requirements to shape
    :return: True if is color image
    """
    return 3 in array.shape


def _is_chw(array: np.ndarray) -> bool:
    """
    Check if channel is first dimension in the array.

    :param array: of shape (x, x, x)
    :return: True of channel is the first dimension
    """
    return array.shape[0] == 3


def simplify_array(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    This function has 3 goals:
        1. convert Tensor to numpy array
        2. if color-image -> transpose to shape (height, width, channel)
        3. if binary image -> squeese to shape (height, width)

    NB! Defined twice in order to avoid circular imports.

    :param image: of arbitrary shape
    :return: array with simplified structure
    """
    image = image.cpu().numpy().squeeze() if isinstance(image, torch.Tensor) else image

    if _is_color_image(image) and _is_chw(image):
        return image.transpose(1, 2, 0)
    elif not _is_color_image(image) and _is_chw(image):
        return image.squeeze()
    return image


def plot_images(axis: bool = True, tight_layout: bool = False, **images):
    """
    Plot images next to each other.

    :param axis: show if True
    :param tight_layout: self-explanatory
    :param images: kwargs as title=image
    """
    image_count = len(images)
    plt.figure(figsize=(image_count * 4, 4))
    for i, (name, image) in enumerate(images.items()):

        plt.subplot(1, image_count, i + 1)
        plt.axis("off") if not axis else None
        # get title from the parameter names
        plt.title(name.replace("_", " ").title(), fontsize=14)
        # plt.imshow(simplify_array(image), cmap="Greys_r")
        plt.imshow(simplify_array(image))
    plt.tight_layout() if tight_layout else None
    plt.show()
