import logging
from collections import Counter
from pathlib import Path
import einops
import torch
from torchvision.transforms import transforms
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2


class RepeatChannelsEinopsd(ImageOnlyTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        # Initialize parent ImageOnlyTransform class
        super(RepeatChannelsEinopsd, self).__init__(always_apply, p)

    def apply(self, image: torch.Tensor, **kwargs):
        # This method is called to apply the transform
        image = einops.repeat(image, "h w -> h w c", c=3)
        return image

    def get_transform_init_args_names(self):
        # This method returns the names of the init args that need to be saved
        # in the serialized representation of the transform (if any).
        # Since this transform does not have any initialization arguments that
        # need to be saved, we return an empty tuple.
        return ()


def prepare_transforms():
    train_augmentations_1 = A.Compose(
        [
            RepeatChannelsEinopsd(),
            A.ToFloat(max_value=254.0),
            A.HorizontalFlip(always_apply=False, p=0.5),
            A.VerticalFlip(always_apply=False, p=0.5),
            A.Rotate(limit=10, always_apply=False, p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                always_apply=False,
                p=0.3,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        seed=42,
    )
    train_augmentations_2 = A.Compose(
        [
            RepeatChannelsEinopsd(),
            A.ToFloat(max_value=254.0),
            A.HorizontalFlip(always_apply=False, p=0.5),
            A.VerticalFlip(always_apply=False, p=0.5),
            A.Rotate(limit=10, always_apply=False, p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                always_apply=False,
                p=0.3,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        seed=42,
    )
    val_augmentations = A.Compose(
        [
            RepeatChannelsEinopsd(),
            A.ToFloat(max_value=254.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        seed=42,
    )
    return [train_augmentations_1, train_augmentations_2], val_augmentations
