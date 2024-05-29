"""
training.py: helper functions for convenient training.

Some of the functions is also used in another project developed by Jan Kokla:
https://github.com/jankokla/epfl_ml-project-2
"""
import os
import torch

import numpy as np
import segmentation_models_pytorch as smp

from collections import defaultdict
from torch.nn import CrossEntropyLoss
from torch import nn
from timm.models import VisionTransformer
from torcheval.metrics.functional import multiclass_f1_score as f1_eval
from pathlib import Path
from tqdm import tqdm


class MetricMonitor:
    """
    Inspired from examples of Albumentation:
        https://albumentations.ai/docs/examples/pytorch_classification/
    """

    def __init__(self, float_precision: int = 3):
        self.float_precision = float_precision
        self.metrics = {}
        self.reset()

    def reset(self):
        """Reset metrics dictionary."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name: str, value):
        """Add value to the metric name."""
        metric = self.metrics[metric_name]

        metric["val"] += value
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def averages(self):
        """Return the average per metric (loss, f1)"""
        return tuple([metric["avg"] for (metric_name, metric) in self.metrics.items()])

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["avg"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def train_epoch_cls(
        model, dataloader, criterion, optimizer, scheduler, epoch
) -> (float, float):
    """
    Train the classification model for one epoch and return epoch loss and
        average f1 score.

    Args:
        model (nn.Module): classification model
        dataloader (torch.Dataloader): used for getting images
        criterion: generally cross-entropy
        optimizer: generally Adam
        scheduler: generally cosine annealing
        epoch: num of epochs to be trained

    Returns:
        averages of metrics (loss and f1)
    """
    device = get_best_available_device()
    model.train()

    metric_monitor = MetricMonitor()
    stream = tqdm(dataloader)

    for i, (inputs, labels, radii) in enumerate(stream, 1):
        inputs, labels, radii = inputs.to(device), labels.to(device), radii.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits = model(inputs.float())

        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred_labels = logits.argmax(dim=-1)
        f1_score = f1_eval(pred_labels, labels, num_classes=16, average="micro")

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("f1", f1_score.item())

        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=(3 - len(str(epoch))) * " " + str(epoch),  # for better alignment,
                metric_monitor=metric_monitor,
            )
        )

    return metric_monitor.averages()


def train_epoch_seg(
        model, dataloader, criterion, optimizer, scheduler, epoch
) -> (float, float):
    """
    Train the segmentation model for one epoch and return epoch loss and
        average f1 score.

    Args:
        model (nn.Module): classification model
        dataloader (torch.Dataloader): used for getting images
        criterion: generally Dice Loss
        optimizer: generally Adam
        scheduler: generally cosine annealing
        epoch: num of epochs to be trained

    Returns:
        averages of metrics (loss and f1)
    """

    is_cls = isinstance(criterion, CrossEntropyLoss)

    device = get_best_available_device()
    model.train()

    metric_monitor = MetricMonitor()
    stream = tqdm(dataloader)

    for i, (inputs, labels, _) in enumerate(stream, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits = model(inputs.float())

        loss = criterion(logits, labels) if is_cls else criterion(logits, labels.float())

        loss.backward()
        optimizer.step()
        scheduler.step()

        if is_cls:
            pred_labels = logits.argmax(dim=-1)
            f1_score = f1_eval(pred_labels, labels, num_classes=16, average="micro")
        else:
            tp, fp, fn, tn = smp.metrics.get_stats(
                logits.sigmoid(), labels, mode="binary", threshold=0.5
            )
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("f1", f1_score.item())

        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=(3 - len(str(epoch))) * " " + str(epoch),  # for better alignment,
                metric_monitor=metric_monitor,
            )
        )

    return metric_monitor.averages()


@torch.no_grad()
def valid_epoch_cls(model, dataloader, criterion, epoch) -> (float, float):
    """
    Validate the model performance by calculating epoch loss and average f1 score.

    :param model: used for inference
    :param dataloader: with validation fold of images
    :param criterion: loss function
    :param epoch: current epoch
    :return: average loss, average f1 score
    """

    device = get_best_available_device()
    model.eval()

    metric_monitor = MetricMonitor()
    stream = tqdm(dataloader)

    for i, (inputs, labels, radii) in enumerate(stream, 1):
        # use gpu whenever possible
        inputs, labels, radii = inputs.to(device), labels.to(device), radii.to(device)

        # predict
        logits = model(inputs.float())

        loss = criterion(logits, labels)

        pred_labels = torch.argmax(logits, dim=-1)
        f1_score = f1_eval(pred_labels, labels, num_classes=17, average="micro")

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("f1", f1_score.item())

        stream.set_description(
            "Epoch: {epoch}. Validation. {metric_monitor}".format(
                epoch=(3 - len(str(epoch))) * " " + str(epoch),  # for better alignment,
                metric_monitor=metric_monitor,
            )
        )

    return metric_monitor.averages()


@torch.no_grad()
def valid_epoch_seg(model, dataloader, criterion, epoch) -> (float, float):
    """
    Validate the model performance by calculating epoch loss and average f1 score.

    :param model: used for inference
    :param dataloader: with validation fold of images
    :param criterion: loss function
    :param epoch: current epoch
    :return: average loss, average f1 score
    """

    device = get_best_available_device()
    model.eval()

    metric_monitor = MetricMonitor()
    stream = tqdm(dataloader)

    for i, (inputs, labels, _) in enumerate(stream, 1):
        # use gpu whenever possible
        inputs, labels = inputs.to(device), labels.to(device)

        # predict
        logits = model(inputs.float())

        # calculate metrics
        loss = criterion(logits, labels.float())

        tp, fp, fn, tn = smp.metrics.get_stats(
            logits.sigmoid(), labels, mode="binary", threshold=0.4
        )
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("f1", f1_score.item())

        stream.set_description(
            "Epoch: {epoch}. Validation. {metric_monitor}".format(
                epoch=(3 - len(str(epoch))) * " " + str(epoch),  # for better alignment,
                metric_monitor=metric_monitor,
            )
        )

    return metric_monitor.averages()


def train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs
) -> tuple:
    """
    Train model for number of epochs and calculate loss and f1.

    :param model: to be trained (with pretrained encoder)
    :param dataloaders: tuple of dataloaders with images (train and validation)
    :param criterion: loss function
    :param optimizer: some SGD implementation
    :param scheduler: for optimizing learning rate
    :param num_epochs:
    :return: lists of train_losses, valid_losses, train_f1s, valid_f1s
    """
    task = model.task
    train_loader, valid_loader = dataloaders

    device = get_best_available_device()
    model.to(device)

    train_losses, valid_losses, train_f1s, valid_f1s = [], [], [], []

    best_valid_loss = float("inf")

    for i in range(num_epochs):

        if task == "classification":
            train_loss, train_f1 = train_epoch_cls(
                model, train_loader, criterion, optimizer, scheduler, i + 1
            )
        else:
            train_loss, train_f1 = train_epoch_seg(
                model, train_loader, criterion, optimizer, scheduler, i + 1
            )

        if valid_loader and task == "classification":
            valid_loss, valid_f1 = valid_epoch_cls(model, valid_loader, criterion, i + 1)

            valid_losses.append(valid_loss)
            valid_f1s.append(valid_f1)

        elif valid_loader:
            valid_loss, valid_f1 = valid_epoch_seg(model, valid_loader, criterion, i + 1)

            valid_f1s.append(valid_f1)
            valid_losses.append(valid_loss)

        train_losses.append(train_loss)
        train_f1s.append(train_f1)

        if valid_loader and round(valid_loss, 3) < best_valid_loss:
            best_valid_loss = round(valid_loss, 3)

            root = Path(__file__).parent.parent
            filepath = f'{root}/models/{task}_{model.coin_type}.pt'
            save_trainable_params(model, filepath)

    return train_losses, valid_losses, train_f1s, valid_f1s


def get_best_available_device() -> str:
    """
    Get best available device for model training.

    Returns:
        best_device (str): prefers CUDA over MPS over CPU
    """
    devices = ['cuda:0', 'mps', 'cpu']
    is_available = [torch.cuda.is_available(), torch.backends.mps.is_available(), True]
    return devices[np.argmax(is_available)]


def find_bn_layers(model: nn.Module, prefix: str, bn_layers: dict):
    """
    Find batch normalization layers from model as we need to save
        running_mean and running_var from them.

    Args:
        model (nn.Module): model that is being considered
        prefix (str): of the param name
        bn_layers (dict): to keep track of them

    Returns:
        bn_layers (dict): with all bn layer names
    """
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.BatchNorm2d):
            # Store the prefix with the layer name
            bn_layers[full_name + '.running_mean'] = child.running_mean
            bn_layers[full_name + '.running_var'] = child.running_var
        # Recurse into child modules
        find_bn_layers(child, full_name if prefix else name, bn_layers)


def save_trainable_params(model: nn.Module, filepath: str) -> None:
    """
    Save only not frozen parameters and batch normalization information.

    Args:
        model (torch.nn.Module): The model whose parameters are to be saved.
        filepath (str): Path to save the filtered state dict.
    """
    # no batch norms yeah
    if isinstance(model.model, VisionTransformer):

        if model.backbone_frozen:
            torch.save(model.model.head.state_dict(), filepath)
        else:
            torch.save(model.model.state_dict(), filepath)

    else:
        bn_layers = {}
        find_bn_layers(model, "", bn_layers)

        filtered_state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                filtered_state_dict[name] = param
        for name, tensor in bn_layers.items():
            filtered_state_dict[name] = tensor

        filtered_state_dict = model.state_dict()

        torch.save(filtered_state_dict, filepath)


def load_params(model, filename: str) -> nn.Module:
    """
    Load parameters from the models directory given the filename

    Args:
        model (torch.nn.Module): The model to load the parameters into
        filename (str): of the .pt file
    """
    root = Path(__file__).parent.parent
    filepath = os.path.join(root, "models", filename)

    saved_state_dict = torch.load(filepath)

    # no batch norms yeah
    if isinstance(model.model, VisionTransformer):

        if model.backbone_frozen:
            model.model.head.load_state_dict(saved_state_dict)
        else:
            model.model.load_state_dict(saved_state_dict)

    else:
        pretrained_dict = dict(model.state_dict())
        # update state dict with saved params
        pretrained_dict.update(saved_state_dict)
        model.load_state_dict(pretrained_dict)

    return model
