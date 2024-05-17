import os
from pathlib import Path

import timm
import torch
from torchvision import models
import segmentation_models_pytorch as smp

from torch import nn
from scripts.training import load_params


class CoinClassifier(nn.Module):
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

        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def forward(self, images):
        return self.model.forward(images)


class HierarchicalClassifier(nn.Module):
    """
    Wrapper model for inference.
    """

    def __init__(self):
        super(HierarchicalClassifier, self).__init__()

        root = Path(__file__).parent.parent

        self.ccy_classifier = CoinClassifier(num_classes=3, coin_type="ccy")
        self.ccy_classifier = load_params(
            self.ccy_classifier,
            os.path.join(root, 'models', 'classification_ccy.pt')
        )

        self.eur_classifier = CoinClassifier(num_classes=8, coin_type="eur")
        self.eur_classifier = load_params(
            self.eur_classifier,
            os.path.join(root, 'models', 'classification_eur.pt')
        )

        self.chf_classifier = CoinClassifier(num_classes=8, coin_type="chf")
        self.chf_classifier = load_params(
            self.chf_classifier,
            os.path.join(root, 'models', 'classification_chf.pt')
        )

    def forward(self, image, radius) -> int:

        logits = self.ccy_classifier(image, radius)
        ccy_id = logits.argmax(dim=-1).squeeze().cpu().numpy().item()

        if ccy_id == 0:
            chf_logits = self.chf_classifier(image, radius)
            return chf_logits.argmax(dim=-1).cpu().numpy().item()

        elif ccy_id == 1:
            eur_logits = self.eur_classifier(image, radius)
            return eur_logits.argmax(dim=-1).cpu().numpy().item() + 7

        else:
            return 15  # OOD class
