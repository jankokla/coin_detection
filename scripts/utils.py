import os

import numpy as np
import torch
import torch.utils.data
import torchvision
from skimage import io, transform
import albumentations as A
from PIL import Image
from pycocotools.coco import COCO
from skimage.color import rgb2hsv
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype
from transformers import ResNetForImageClassification


def read_image(
        filename: str,
        new_height: int = 400,
        new_width: int = 600
) -> np.ndarray:
    image = io.imread(f'data/train/{filename}')
    resized = transform.resize(image, (new_height, new_width), anti_aliasing=True)

    return (resized * 255).astype(np.uint8)


def filter_between_lines(hue_image, saturation_image):
    # Define the parameters for the first line
    slope1 = 8.9
    intercept1 = 0.2

    # Define the parameters for the second line
    slope2 = 3
    intercept2 = 0.7

    # Calculate y values for both lines across the hue image
    y_line1 = slope1 * hue_image.astype(np.float32) + intercept1
    y_line2 = slope2 * hue_image.astype(np.float32) + intercept2

    # Create a mask for pixels between the two lines
    # Note: Ensure y_line1 is the upper line and y_line2 is the lower line,
    # if it's not the case, swap them.
    between_mask = (saturation_image > np.minimum(y_line1, y_line2)) & (saturation_image < np.maximum(y_line1, y_line2))

    # Result where 1 indicates pixels between lines, 0 otherwise
    result = np.where(between_mask, 0, 1)

    return result


def check_position(hue_image, saturation_image):
    # Define the line parameters
    slope = 8.8889
    y_intercept = 0.2

    # Calculate y_line for every hue value at once
    y_line = slope * hue_image.astype(np.float32) + y_intercept

    # Compare saturation image to y_line
    mask = np.where(saturation_image < y_line, 1, 0)  # Above the line

    return mask


def apply_hsv_threshold(img):
    """
    Apply threshold to the input image in hsv colorspace.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """
    # Use the previous function to extract HSV channels
    data_h, data_s, data_v = extract_hsv_channels(img=img)

    mask = filter_between_lines(data_h, data_s)

    hue_thresh = 0.3
    value_thresh = 0.95

    mask = np.logical_and(
        data_h < hue_thresh,
        mask
    )

    mask = np.logical_or(mask, value_thresh > 0.95)

    img[~mask] = (255, 255, 255)

    return img


class myOwnDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotation=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    @staticmethod
    def _bbox_to_pytorch(bboxes) -> torch.tensor:
        """
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]

        :param annotations:
        :return:
        """
        new_bboxes = []
        for bbox in bboxes:
            x_min, y_min = bbox[0], bbox[1]
            x_max, y_max = x_min + bbox[2], y_min + bbox[3]
            new_bboxes.append([x_min, y_min, x_max, y_max])
        return new_bboxes

    def __getitem__(self, index):

        # set up coco
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        bboxes = [d['bbox'] for d in coco_annotation]
        labels = [d['category_id'] for d in coco_annotation]

        # get image
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path))

        if self.transforms is not None:
            transformed = self.transforms(image=np.array(img), bboxes=bboxes, class_labels=labels)
            img = torch.from_numpy(transformed['image']).permute(2, 0, 1)
            bboxes = transformed['bboxes']
            labels = transformed['class_labels']

        bboxes = self._bbox_to_pytorch(bboxes)

        target = {
            "boxes": torch.tensor(bboxes).to(dtype=torch.int64),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(img_id, dtype=torch.int64)
        }

        return convert_image_dtype(img), target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    """Necessary for batches."""
    return tuple(zip(*batch))


def _get_backbone(backbone: str) -> nn.Module:
    """
    Get backbone based on input string.

    Args:
        backbone (str): could be ["mobile_net", "resnet18", "resnet50"]

    Returns:
        backbone (nn.Module): backbone with frozen weights
    """
    if backbone == "mobile_net":
        backbone = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2").features
        backbone.out_channels = 1280

    elif backbone == "resnet18":
        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V2")
        modules = list(backbone.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512

    elif backbone == "resnet50":
        backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        modules = list(backbone.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048

    else:
        raise Exception("Unsupported backbone type.")

    # freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    return backbone


def get_model_coco(num_classes: int = 17):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_object_detection_model(backbone: str, num_classes: int = 17):
    """
    Return FasterRCNN object with specified backbone and head.

    Args:
        backbone (str): name of the backbone, see helper function.
        num_classes (int): how many coins + background as 0

    Returns:
        model (FasterRCNN): with frozen backbone
    """
    backbone: nn.Module = _get_backbone(backbone=backbone)

    # make the RPN generate only boxes since we have coins
    anchor_generator = AnchorGenerator(
        sizes=((64, 128, 256),),
        aspect_ratios=((1.0, 1.0, 1.0),)
    )

    # pooler with default parameters
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # put the pieces together inside a Faster-RCNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_nms_thresh=0.5
    )

    return model


def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """
    img_hsv = rgb2hsv(np.copy(img))
    data_h = img_hsv[:, :, 0]
    data_s = img_hsv[:, :, 1]
    data_v = img_hsv[:, :, 2]

    return data_h, data_s, data_v