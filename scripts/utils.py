import json
import os
import warnings
from functools import partial
from typing import List, Tuple, Union
import cv2 as cv

import numpy as np
import torch.utils.data
from PIL import Image
from matplotlib import image as mpimg
import albumentations as A
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch import nn

from scripts.config import id_to_label
from scripts.training import get_best_available_device


class ClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset class for segmentation.

    Args:
        image_paths (List[str]): list of full paths to images
        labels_path (List[str]): path to classification JSON
        transform (A.Compose): custom transformations from Albumentations
        preprocess (partial): encoder-specific transforms callable
    """

    def __init__(
            self,
            image_paths: Union[List[str], np.ndarray],
            labels_path: str = None,
            transform: A.Compose = None,
            preprocess: partial = None
    ):

        self.image_paths = image_paths

        if labels_path:
            with open(labels_path, 'r') as file:
                self.labels = json.load(file)
        else:
            self.labels = None

        self.transform = transform
        self.preprocess = preprocess

    def __getitem__(self, i):

        if isinstance(self.image_paths, np.ndarray):
            image = self.image_paths
            label = -1
        else:
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

        return (torch.tensor(image, dtype=torch.float32),
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

        self.images = image_paths
        self.masks = mask_paths if mask_paths else None

        self.transform = transform
        self.preprocess = preprocess

    def __getitem__(self, i):

        filename = self.images[i].split('/')[-1]
        image = mpimg.imread(self.images[i])

        # if no mask use dummy mask
        mask = (
            np.where(mpimg.imread(self.masks[i]) > 0, 1, 0).astype(np.uint8)
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

        return image, mask, filename

    def __len__(self):
        return len(self.images)


@torch.no_grad()
def get_segmentation(model: nn.Module, image: torch.Tensor) -> np.ndarray:
    """
    Return segmentation prediction for the specific image.

    Args:
        model (nn.Module): must be trained
        image (torch.Tensor): for which we need to predict the mask

    Returns:
        segmentation (np.ndarray): mask with values in {0, 255}
    """
    device = get_best_available_device()
    image, model = image.to(device), model.to(device)
    model.eval()

    logits = model(image.float())
    prediction_sigmoid = logits.sigmoid().cpu().numpy().squeeze()
    bool_array = np.where(prediction_sigmoid >= 0.5, 1, 0)

    return (bool_array * 255).astype(np.uint8)


@torch.no_grad()
def get_class(model: nn.Module, coin, radius) -> str:
    """
    Return class for the specific coin.

    Args:
        model (nn.Module): must be trained
        coin (torch.Tensor): image of the coin
        radius (torch.Tensor): of the coin

    Returns:
        class_label (str): coin name
    """
    device = get_best_available_device()
    model, coin, radius = model.to(device), coin.to(device), radius.to(device)

    logits = model(coin, radius)
    pred = id_to_label(logits.argmax(dim=-1).cpu().numpy())

    return pred[0]


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
    Split data to training and validation.

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
    elif type == 'classification':
        image_paths, mask_paths =_get_paths_classification(images_path)
    else:
        image_paths = [
            os.path.join(images_path, image)
            for image in sorted(os.listdir(images_path))
        ]
        mask_paths = None

    # All images in train set, none in test
    if type == 'inference':
        return image_paths, [], [], []
    elif test_size == 0:
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


def get_images_from_coco(images_path: str, annotation_json: str) -> None:
    """
    Based on coco JSON file cut coins from images and save them to files.

    Args:
        images_path (str): original images
        annotation_json (str): coco JSON path
    """
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


def generate_hough(
        prediction: np.ndarray,
        original_img: Union[np.ndarray, torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Hough circles and plot them to image.

    Args:
        prediction (np.ndarray): binary image
        original_img (Union[np.ndarray, torch.Tensor]): as image template

    Returns:
        circles (np.ndarray): list of tuples as (x, y, r)
        hough_img (np.ndarray): image with Hough circles
    """
    circles = cv.HoughCircles(
        prediction, cv.HOUGH_GRADIENT, 1, 20,
        param1=200, param2=10, minRadius=15, maxRadius=55
    )
    circles = filter_circles(circles)

    if isinstance(original_img, torch.Tensor):
        original_img = original_img.cpu().numpy()

    hough_img = original_img.copy()

    # if comes as a batch (1, 3, H, W)
    if len(hough_img.shape) == 4:
        hough_img = hough_img.squeeze(0)

    # if channel is first -> HWC
    if hough_img.shape[0] == 3:
        hough_img = hough_img.transpose(1, 2, 0)

    hough_img = cv.cvtColor(hough_img, cv.COLOR_BGR2RGB)

    for (x, y, r) in circles:
        cv.circle(hough_img, (x, y), r, (255, 0, 0), 4)

    return circles, hough_img


def get_cropped_image(
        image: np.ndarray, x, y, r, x_ratio, y_ratio, padding: int = 40
) -> np.ndarray:
    """
    Based on circle info and ratio, crop the coin from the image.

    Args:
        image (np.ndarray): from where to cut the coins
        x (float): center x-coordinate
        y (float): center y-coordinate
        r (float): radius
        x_ratio (float): scaling factor for circle info
        y_ratio (float): scaling factor for circle info
        padding (int): add some margin on the edges

    Returns:
        cropped_image (np.ndarray)
    """
    x = int(x * x_ratio)
    y = int(y * y_ratio)
    r = int(r * max(x_ratio, y_ratio))

    # calculate the region of box
    top_left_x = max(0, x - r - padding)
    top_left_y = max(0, y - r - padding)
    bottom_right_x = x + r + padding
    bottom_right_y = y + r + padding

    # crop the image
    return image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
