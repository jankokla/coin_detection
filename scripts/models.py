"""models.py: model classes for training and inference."""
import os
import random
import timm
import torch

import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

from pathlib import Path
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader

from scripts.training import load_params, get_best_available_device
from scripts.utils import get_segmentation, generate_hough, get_cropped_image, \
    get_bb_coordinates, ClassificationDataset, get_class
from scripts.config import ID_TO_LABEL, ID_TO_CCY, ID_TO_EUR, \
    ID_TO_SIDE, ID_TO_CHF_IMG, row_template, example_row, SIZE_DICT


class CoinClassifier(nn.Module):
    """
    Unified model wrapper for all classification models that is compatible
        with the training pipeline.

    Args:
        num_classes (int): output size of the final linear layer
        coin_type (str): used only for logging / saving purposes
        freeze (bool): if True -> freeze the backbone
    """

    task = "classification"

    def __init__(
            self,
            num_classes: int = 16,
            coin_type: str = "",
            freeze: bool = True
    ):

        super().__init__()

        self.coin_type = coin_type

        self.model = timm.create_model(
            'vit_small_patch16_224.augreg_in1k',
            pretrained=True,
            num_classes=num_classes
        )

        self.backbone_frozen = freeze

        if freeze:

            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.head.parameters():
                param.requires_grad = True

    def forward(self, images):
        output = self.model(images)
        return output


class CoinLocalizer(nn.Module):
    """
    Wrapper for segmentation model that makes it compatible with unified
        training pipeline.

    Args:
        encoder_name (str): backbone that will be used for feature extraction
        num_classes (int): binary segmentation -> 1 class
        num_channels (int): if RGB -> 3
    """
    task = "segmentation"
    coin_type = ""

    def __init__(
            self,
            encoder_name: str = "resnet50",
            num_classes: int = 1,
            num_channels: int = 3
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=num_channels,
            classes=num_classes,
        )

        # freeze encoder parameters
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def forward(self, images):
        return self.model.forward(images)


class HierarchicalClassifier(nn.Module):
    """
    Wrapper model that includes all separate classification models:
        CCY model
        ├── EUR model
        └── CHF head-tail model
            ├── CHF tail model
            └── CHF head model
    """

    def __init__(self):
        super(HierarchicalClassifier, self).__init__()

        root = Path(__file__).parent.parent

        # initialize segmentation model
        self.seg_model = CoinLocalizer()
        seg_path = os.path.join(root, 'models', 'segmentation_.pt')
        self.seg_model = load_params(self.seg_model, seg_path)

        # freeze its parameters (if we try to do some final fine-tuning)
        for param in self.seg_model.parameters():
            param.requires_grad = False

        # initialize ccy model: predicts EUR, CHF
        self.ccy_model = CoinClassifier(num_classes=2, coin_type="ccy")
        ccy_path = os.path.join(root, 'models', 'classification_ccy.pt')
        self.ccy_model = load_params(self.ccy_model, ccy_path)

        # initialize EUR model: predicts EUR coin types
        self.eur_model = CoinClassifier(num_classes=8, coin_type="eur", freeze=False)
        eur_path = os.path.join(root, 'models', 'classification_eur.pt')
        self.eur_model = load_params(self.eur_model, eur_path)

        # initialize "side" model: predicts if it's head or tails (CHF)
        self.side_model = CoinClassifier(num_classes=2, coin_type="heads-tails")
        side_path = os.path.join(root, 'models', 'classification_heads-tails.pt')
        self.side_model = load_params(self.side_model, side_path)

        # initialize CHF tails model: predicts CHF coin type from tail image
        self.chf_tail_model = CoinClassifier(num_classes=7, coin_type="chf-tails", freeze=False)
        chf_tail_path = os.path.join(root, 'models', 'classification_chf-tails.pt')
        self.chf_tail_model = load_params(self.chf_tail_model, chf_tail_path)

        # initialize CHF heads model: predicts CHF coin type based on the head
        self.chf_head_model = CoinClassifier(num_classes=3, coin_type="chf-heads", freeze=False)
        chf_head_path = os.path.join(root, 'models', 'classification_chf-heads.pt')
        self.chf_head_model = load_params(self.chf_head_model, chf_head_path)

        # define label mappers
        self.id_to_label = np.vectorize(lambda x: ID_TO_LABEL.get(x, "unknown"))
        self.id_to_ccy = np.vectorize(lambda x: ID_TO_CCY.get(x, "unknown"))
        self.id_to_eur = np.vectorize(lambda x: ID_TO_EUR.get(x, "unknown"))
        self.id_to_side = np.vectorize(lambda x: ID_TO_SIDE.get(x, "unknown"))
        self.id_to_chf_img = np.vectorize(lambda x: ID_TO_CHF_IMG.get(x, "unknown"))

        # specify OOD threshold (if confidence is lower -> OOD)
        self.ccy_confidence_thresh = nn.Parameter(torch.tensor(0.75))
        self.eur_confidence_thresh = nn.Parameter(torch.tensor(0.6))

        # resize and apply imagenet-specific tf for coin classification
        self.cls_tf = A.Compose([
            A.Resize(width=224, height=224, always_apply=True),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                always_apply=True
            )
        ])

        self.device = get_best_available_device()

    @staticmethod
    def make_final_decision(
            text_labels: list,
            probabilities: list,
            radii: list
    ) -> list:
        """
        As we have some CHF coins that have same head, we can only infer their
            correct value if we compare them with the coins on the image
            that we're really certain of. This function does exactly that.

        Args:
            text_labels (list): strings that is used by the project definition,
                additionally undefined ones:
                    - '2CHF/1CHF/0.5CHF'
                    - '0.2CHF/0.1CHF/0.05CHF'
            probabilities (list): how sure the model is about the coin value
            radii (list): corresponding radii values

        Returns:
            text_labels (list): updated version, where we have replaced the
                doubtful ones ('2CHF/1CHF/0.5CHF', '0.2CHF/0.1CHF/0.05CHF')
                with our best guess
        """
        unknown_labels = ['2CHF/1CHF/0.5CHF', '0.2CHF/0.1CHF/0.05CHF']
        chf_idx = [i for i, text_label in enumerate(text_labels) if text_label in unknown_labels]
        ood_idx = [i for i, text_label in enumerate(text_labels) if text_label in ["OOD"]]

        remaining_prob_with_idx = [
            (i, p) for i, p in enumerate(probabilities) if (i not in chf_idx and i not in ood_idx)
        ]
        remaining_prob_values = [p for i, p in remaining_prob_with_idx]

        if remaining_prob_values:

            # get the index of the coin with the highest probability
            max_prob_idx = np.argmax(remaining_prob_values)

            # get the original index of the highest probability
            max_prob_original_idx = remaining_prob_with_idx[max_prob_idx][0]

            for index in chf_idx:

                coin_radii = radii[index]
                refer_coin_radii = radii[max_prob_original_idx]
                refer_coin_label = text_labels[max_prob_original_idx]
                real_ratio = refer_coin_radii / coin_radii

                if text_labels[index] == '2CHF/1CHF/0.5CHF':

                    # calculate what ratios we should see for these coins
                    refer_ratio_2chf = SIZE_DICT[refer_coin_label] / SIZE_DICT['2CHF']
                    refer_ratio_1chf = SIZE_DICT[refer_coin_label] / SIZE_DICT['1CHF']
                    refer_ratio_05chf = SIZE_DICT[refer_coin_label] / SIZE_DICT['0.5CHF']

                    # get the closest one and replace the label value in list
                    current_labels = ['2CHF', '1CHF', '0.5CHF']
                    ratios = [refer_ratio_2chf, refer_ratio_1chf, refer_ratio_05chf]
                    closest_index = min(enumerate(ratios), key=lambda x: abs(x[1] - real_ratio))[0]

                    text_labels[index] = current_labels[closest_index]

                else:   # '0.2CHF/0.1CHF/0.05CHF'

                    # calculate what ratios we should see for these coins
                    refer_ratio_02chf = SIZE_DICT[refer_coin_label] / SIZE_DICT['0.2CHF']
                    refer_ratio_01chf = SIZE_DICT[refer_coin_label] / SIZE_DICT['0.1CHF']
                    refer_ratio_005chf = SIZE_DICT[refer_coin_label] / SIZE_DICT['0.05CHF']

                    # get the closest one and replace the label value in list
                    current_labels = ['0.2CHF', '0.1CHF', '0.05CHF']
                    ratios = [refer_ratio_02chf, refer_ratio_01chf, refer_ratio_005chf]
                    closest_index = min(enumerate(ratios), key=lambda x: abs(x[1] - real_ratio))[0]

                    text_labels[index] = current_labels[closest_index]

        # if undefined coin is the only one on the image -> pick randomly
        else:
            for index in chf_idx:
                if text_labels[index] == '2CHF/1CHF/0.5CHF':
                    text_labels[index] = random.choice(['2CHF', '1CHF', '0.5CHF'])
                else:
                    text_labels[index] = random.choice(['0.2CHF', '0.1CHF', '0.05CHF'])

        return text_labels

    def forward(self, image, original_img) -> tuple:

        # get segmentation and apply Hough
        predicted = get_segmentation(self.seg_model, image)
        circles, hough_img = generate_hough(predicted, image)

        # segmentation was done on smaller images -> reset the coordinates for original images
        x_ratio = 6000 / image.shape[3]
        y_ratio = 4000 / image.shape[2]

        labels = []
        probabilities = []
        boxes = []
        radii = [r for x, y, r in circles]

        for j, (x, y, r) in enumerate(circles):

            cropped_image = get_cropped_image(original_img, x, y, r, x_ratio, y_ratio)

            box = list(get_bb_coordinates(x, y, r, x_ratio, y_ratio))
            boxes.append(box)

            # initiate the dataloader
            coin_loader = DataLoader(
                ClassificationDataset(cropped_image, transform=self.cls_tf)
            )
            coin_iterator = iter(coin_loader)
            coin, _, radius = next(coin_iterator)

            coin = coin.to(self.device)

            # predict currency
            ccy_id, ccy_label, ccy_prob = get_class(self.ccy_model, coin, self.id_to_ccy)

            # if not sure about the currency -> OOD
            if ccy_prob < self.ccy_confidence_thresh:
                labels.append("OOD")
                probabilities.append(ccy_prob)

            # if EUR -> predict EUR coin type
            elif ccy_id == 1:

                eur_id, eur_label, eur_prob = get_class(
                    self.eur_model, coin, self.id_to_eur
                )

                if eur_prob < self.eur_confidence_thresh:
                    labels.append("OOD")
                else:
                    labels.append(f"{eur_label}")

                probabilities.append(eur_prob)

            # if CHF -> predict CHF head or tails
            else:

                side_id, side_label, side_prob = get_class(
                    self.side_model, coin, self.id_to_side
                )

                # if tail -> predict CHF coin type
                if side_id == 0:
                    chf_tail_id, chf_tail_label, chf_tail_prob = get_class(
                        self.chf_tail_model, coin, self.id_to_label
                    )
                    labels.append(f"{chf_tail_label}")
                    probabilities.append(chf_tail_prob)

                # if head -> predict picture type (3 options)
                else:
                    chf_head_id, chf_head_label, chf_head_prob = get_class(
                        self.chf_head_model, coin, self.id_to_chf_img
                    )

                    # since 5CHF has distinctive picture
                    if chf_head_id == 0:
                        labels.append("5CHF")

                    else:
                        labels.append(f"{chf_head_label}")

                    probabilities.append(chf_head_prob)

        labels = self.make_final_decision(labels, probabilities, radii)

        # generate row for the final csv
        row = row_template.copy()
        counts = dict(Counter(labels))
        row.update(counts)
        count_list = [row[coin] for coin in example_row]

        return (
            torch.tensor(count_list, dtype=torch.float32, requires_grad=True),
            torch.tensor(boxes, dtype=torch.int16),
            labels
        )
