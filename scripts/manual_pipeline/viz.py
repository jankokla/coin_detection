from typing import Union, Callable

import numpy as np
import torch
from PIL import ImageDraw
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image


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


# Plot color space distribution
def plot_colors_histo(
        img: np.ndarray,
        func: Callable,
        labels: list[str],
):
    """
    Plot the original image (top) as well as the channel's color distributions (bottom).

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    func: Callable
        A callable function that extracts D channels from the input image
    labels: list of str
        List of D labels indicating the name of the channel
    """

    # Extract colors
    channels = func(img=img)
    C2 = len(channels)
    M, N, C1 = img.shape
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, C2)

    # Use random seed to downsample image colors (increase run speed - 10%)
    mask = np.random.RandomState(seed=0).rand(M, N) < 0.1

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Plot channel distributions
    ax1.scatter(channels[0][mask].flatten(), channels[1][mask].flatten(), c=img[mask] / 255, s=1, alpha=0.1)
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_title("{} vs {}".format(labels[0], labels[1]))
    ax2.scatter(channels[0][mask].flatten(), channels[2][mask].flatten(), c=img[mask] / 255, s=1, alpha=0.1)
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[2])
    ax2.set_title("{} vs {}".format(labels[0], labels[2]))
    ax3.scatter(channels[1][mask].flatten(), channels[2][mask].flatten(), c=img[mask] / 255, s=1, alpha=0.1)
    ax3.set_xlabel(labels[1])
    ax3.set_ylabel(labels[2])
    ax3.set_title("{} vs {}".format(labels[1], labels[2]))

    plt.tight_layout()