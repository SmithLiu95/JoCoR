import cv2
import torchvision.transforms as T
import numpy as np
from PIL import Image

def build_transforms(is_train=True):
    PIXEL_MEAN= [0.485, 0.456, 0.406]
    PIXEL_STD= [0.229, 0.224, 0.225]
    ORIGIN_SIZE = [256, 256]
    CROP_SCALE = [0.2, 1]
    CROP_SIZE = [224, 224]
    normalize_transform = T.Normalize(
        mean=PIXEL_MEAN, std=PIXEL_STD
    )
    if is_train:
        transform = T.Compose(
            [
                T.Resize(size=ORIGIN_SIZE[0]),
                T.RandomResizedCrop(
                    scale=CROP_SCALE, size=CROP_SIZE[0]
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize(size=ORIGIN_SIZE[0]),
                T.CenterCrop(CROP_SIZE[0]),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform