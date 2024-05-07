import json
import os
from functools import partial
from typing import List, Tuple

import numpy as np
import torch.utils.data
from PIL import Image
from matplotlib import image as mpimg
import albumentations as A
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

from scripts.training import get_best_available_device


class ClassificationDataset(torch.utils.data.Dataset):
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
            labels_path: str = None,
            transform: A.Compose = None,
            preprocess: partial = None
    ):

        self.image_paths = image_paths

        with open(labels_path, 'r') as file:
            self.labels = json.load(file)

        self.transform = transform
        self.preprocess = preprocess

    def __getitem__(self, i):

        filepath = self.image_paths[i]
        image = mpimg.imread(filepath)
        label = self.labels[filepath.split('/')[-1]]

        radius = np.mean(image.shape[:2]) / 2

        if self.transform:
            # apply same transformation to image and mask
            # NB! This must be done before converting to Pytorch format
            transformed = self.transform(image=image)
            image = transformed["image"]

        # apply preprocessing to adjust to encoder
        if self.preprocess:
            sample = self.preprocess(image)
            image = sample["image"]

        # convert to Pytorch format HWC -> CHW
        image = np.moveaxis(image, -1, 0)

        return (torch.tensor(image),
                torch.tensor(label),
                torch.tensor(radius, dtype=torch.float32))

    def __len__(self):
        return len(self.image_paths)


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
            np.where(self.masks[i] > 0, 1, 0).astype(np.uint8)
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
            sample = self.preprocess(image, mask)
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
    Return segmentation prediction for the specific image.

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


def _get_paths_segmentation(images_path: str) -> Tuple[list, list]:
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

    return image_paths, mask_paths


def _get_paths_classification(images_path: str) -> Tuple[list, None]:
    # List to hold file names
    image_paths = []

    for file in os.listdir(images_path):
        if file.endswith('.jpg'):
            image_paths.append(os.path.join(images_path, file))

    return image_paths, None


def split_data(images_path: str, test_size: float, type: str):
    """

    Args:
        images_path (str): absolute or relative path of the img directory
        test_size (float): from range [0, 1]
        type (str): either "segmentation" or "classification"

    Returns:
        image_path_train (List[str])
        image_path_test (List[str])
        mask_path_train (List[str])
        mask_path_test (List[str])
    """
    if type == 'segmentation':
        image_paths, mask_paths = _get_paths_segmentation(images_path)
    else:
        image_paths, mask_paths =_get_paths_classification(images_path)

    # All images in train set, none in test
    if test_size == 0:
        return image_paths, [], mask_paths, []
    elif type == 'classification':
        train, test = train_test_split(image_paths, test_size=test_size)
        return train, test, None, None
    else:
        return train_test_split(image_paths, mask_paths, test_size=test_size)


def filter_circles(hough_output: np.ndarray) -> np.ndarray:
    """
    If Hough Transform returns circles that overlap each other,
        filter them out and keep only the biggest circle to make
        sure that the coin is fully covered.

    Args:
        hough_output (np.ndarray): with shape (1, N, 3) or shape(N, 3)

    Returns:
        filtered_output (np.ndarray): with shape (K, 3)
    """
    # if shape is not (N, 3) make it so
    if len(hough_output.shape) != 2:
        hough_output = hough_output.squeeze(0)

    # make sure you have uint16 as dtype for cropping and plotting
    if hough_output.dtype != np.dtype('uint16'):
        hough_output = np.uint16(np.around(hough_output))

    # extract centers and radii
    centers, radii = hough_output[:, :2], hough_output[:, 2]

    distances = pairwise_distances(centers[:, :2])

    # get call the overlapping circles and clean bottom half
    is_inside = distances < radii
    is_inside[np.tril_indices(len(is_inside), 0)] = False

    # iterate over indices to find what to keep
    keep = np.full(len(hough_output), True)
    for i in range(len(hough_output)):

        if not keep[i]:
            continue

        # find all circles where i's center is inside and i is not the largest
        overlapping = is_inside[:, i]
        larger = radii[i] > radii[overlapping]
        if not all(larger):
            keep[i] = False

        # keep only the biggest circle
        keep[overlapping & (radii[i] >= radii)] = False

    return hough_output[keep]


def get_images_from_coco(images_path, annotation_json) -> None:

    # if data already downloaded -> no need for action
    if len(os.listdir('data/classification')) != 0:
        print('Files already there, good to go!')
        return

    # load the json file
    with open(annotation_json, 'r') as file:
        data = json.load(file)

    # map image id to file name
    annotations = data['annotations']
    images = {img['id']: img['file_name'] for img in data['images']}

    # create the output list
    labels = {}

    # get the annotations
    for ann in annotations:
        image_id = ann['image_id']
        filename = images.get(image_id)

        image_path = os.path.join(images_path, filename)
        image = Image.open(image_path)

        # get the bounding box
        x_min, y_min, width, height = ann['bbox']

        # calculate the bounding box coordinates
        left, right = int(x_min), int(x_min + width)
        top, bottom = int(y_min), int(y_min + height)

        # crop the image
        cropped_image = image.crop((left, top, right, bottom))

        # save the cropped image
        cropped_name = f"{filename.rsplit('.', 1)[0]}_cropped_{ann['id']}.jpg"
        cropped_image.save(os.path.join('data/classification', cropped_name))

        labels[cropped_name] = ann['category_id']

    with open("data/classification/labels.json", "w") as outfile:
        json.dump(labels, outfile)


