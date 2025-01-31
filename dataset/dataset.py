from collections import Counter
from skimage import io
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import os

import torch
from torch.utils.data import Dataset


class CSAWM(Dataset):
    def __init__(self, data, data_dir, num_classes, transform, loss):
        self.data_dir = Path(data_dir)
        self.data = data
        self.transform = transform
        self.num_classes = num_classes
        self.targets = self.data["Label"].to_numpy()
        self.loss = loss
        self.class_num_list = self.get_num_class_list()

    def get_num_class_list(self):
        if self.loss == "ordinal_encoding":
            adjusted_num_classes = self.num_classes - 1
            multi_hot_labels = [self.get_multihot(label) for label in self.targets]
            multi_hot_tuples = [tuple(label.tolist()) for label in multi_hot_labels]
            counter = Counter(multi_hot_tuples)
            class_num = [
                counter[tuple([1] * i + [0] * (adjusted_num_classes - i))]
                for i in range(adjusted_num_classes)
            ]
            return class_num
        else:
            class_num = np.array([0] * self.num_classes)
            for label in self.targets:
                class_num[label] += 1
            return class_num.tolist()

    def get_multihot(self, label):
        multi_hot = [0] * (self.num_classes - 1)
        label = label - 1
        if label > 0:
            for i in range(label):
                multi_hot[i] = 1
        return torch.tensor(multi_hot, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = str(self.data_dir / row["Filename"])

        # Load the image
        original_image = io.imread(image_path)

        # Process data
        training_data = {
            "image": original_image,
            "label": int(row["Label"]),
        }

        # Apply transform to training data
        if self.transform:
            if not isinstance(self.transform, list):
                img_dict = self.transform(image=training_data["image"])
                img = img_dict["image"]
            else:
                img0_dict = self.transform[0](image=training_data["image"])
                img0 = img0_dict["image"]
                img1_dict = self.transform[1](image=training_data["image"])
                img1 = img1_dict["image"]
                img = [img0, img1]

        return img, training_data["label"]

    def __len__(self):
        return len(self.data)
