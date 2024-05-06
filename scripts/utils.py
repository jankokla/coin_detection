import os
from functools import partial
from typing import List

import numpy as np
import torch.utils.data
from matplotlib import image as mpimg
import albumentations as A
from sklearn.model_selection import train_test_split

from scripts.training import get_best_available_device


class SegmentationDataset(torch.utils.data.Dataset):
    """
    Dataset class for segmentation.

    Args:
        image_paths (List[str]): list of full paths to images
        mask_paths (List[str]): list of full paths to masks
        transform (A.Compose): custom transformations from Albumentations
        preprocess (partial): encoder-specific transforms callable
    """

    def __init__(
            self,
            image_paths: List[str],
            mask_paths: List[str] = None,
            transform: A.Compose = None,
            preprocess: partial = None
    ):

        self.images = [mpimg.imread(path) for path in image_paths]
        self.masks = [mpimg.imread(path) for path in mask_paths] if mask_paths else None

        self.transform = transform
        self.preprocess = preprocess

    def __getitem__(self, i):

        image = self.images[i]
        # if no mask use dummy mask
        mask = (
            np.where(self.masks[i] >= 0.5, 1, 0).astype(np.uint8)
            if self.masks
            else np.zeros(image.shape)
        )

        if self.transform:
            # apply same transformation to image and mask
            # NB! This must be done before converting to Pytorch format
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # apply preprocessing to adjust to encoder
        if self.preprocess:
            sample = self.preprocess(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # convert to Pytorch format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)

        return image, mask

    def __len__(self):
        return len(self.images)


@torch.no_grad()
def get_prediction(model, image) -> np.ndarray:
    """
    Return prediction for the specific image.

    :param model: used for inference
    :param image: torch.Tensor
    :return: segmented image
    """
    device = get_best_available_device()
    image = image.to(device)
    model.eval()
    logits = model(image.float())
    prediction_sigmoid = logits.sigmoid().cpu().numpy().squeeze()
    return np.where(prediction_sigmoid >= 0.5, 1, 0)


def split_data(images_path: str, test_size: float):
    """

    Args:
        images_path (str): absolute path of the parent directory of images
        test_size (float): from range [0, 1]

    Returns:
        image_path_train (List[str])
        image_path_test (List[str])
        mask_path_train (List[str])
        mask_path_test (List[str])
    """
    # specify image and ground truth full path
    image_directory = os.path.join(images_path, "images")
    labels_directory = os.path.join(images_path, "masks")

    # specify absolute paths for all files
    image_paths = [
        os.path.join(image_directory, image)
        for image in sorted(os.listdir(image_directory))
    ]
    mask_paths = [
        os.path.join(labels_directory, image)
        for image in sorted(os.listdir(labels_directory))
    ]

    # All images in train set, none in test
    if test_size == 0:
        return image_paths, [], mask_paths, []
    else:
        return train_test_split(image_paths, mask_paths, test_size=test_size)
