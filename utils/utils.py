import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing import List
import torch.nn as nn
import pandas as pd
import yaml
from pathlib import Path
from dlordinal.losses import BinomialCrossEntropyLoss

from utils.losses import OrdinalEncodingLoss


def parse_yaml(path_to_yaml: Path) -> dict:
    with open(path_to_yaml, "r") as file:
        return yaml.safe_load(file)


def get_imageids_and_labels_per_centre(
    df: pd.DataFrame, fold_num: int, alpha: float, p: float
):
    """
    Retrieves imges IDs and their corresponding labels grouped by centre for a given fold, alpha, and p combination.
    """

    # Construct the column name based on the fold, alpha and p
    column_name = f"Fold_{fold_num}_Alpha_{alpha}_P_{p}"

    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataset.")

    # Group by centre (based on column_name) and retrieve filenames and labels
    image_ids_labels_per_centre = (
        df.groupby(column_name)
        .apply(lambda x: list(zip(x["Filename"], x["Label"])))
        .to_dict()
    )

    return image_ids_labels_per_centre


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_class_num(ys):
    from collections import Counter

    num_dict = Counter(ys)
    index = []
    compose = []
    for c in num_dict.keys():
        if num_dict[c] != 0:
            index.append(c)
            compose.append(num_dict[c])
    return index, compose


def get_num_classes(loss, num_classes):
    """
    Get the number of classes based on the loss function.
    """
    class_modifiers = {
        "ce": 0,
        "binomial": 0,
        "ordinal_encoding": -1,
    }

    return num_classes + class_modifiers.get(loss, 0)


def get_criterion(
    criterion_name: str, num_classes: int, device: torch.device, per_class=False
):
    """
    Return the appropriate loss function based on criterion_name.
    """
    if per_class == True:
        reduction = "none"
    else:
        reduction = "mean"
    criterion_map = {
        "ce": nn.CrossEntropyLoss(reduction=reduction),
        "binomial": BinomialCrossEntropyLoss(
            num_classes=num_classes, reduction=reduction
        ),
        "ordinal_encoding": OrdinalEncodingLoss(num_classes=num_classes),
    }

    if criterion_name not in criterion_map:
        raise ValueError(
            f"Unknown criterion: {criterion_name}. The available criterions are: {criterion_map.keys}"
        )

    return criterion_map[criterion_name]


def classify_label(y, num_classes: int):
    """Function that returns a list of as many lists as classes present in the train set.
    In each list, we have the indees of the samples with the label"""
    list1: List[List[int]] = [[] for _ in range(num_classes)]
    for idx, label in enumerate(y):
        list1[int(label)].append(idx)
    return list1


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
