import numpy as np
import random
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
from PIL import Image
from ..util.constants import pvnet_mean, pvnet_std

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None, mask=None):
        for t in self.transforms:
            img, kpts, mask = t(img, kpts, mask)
        return img, kpts, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    
    def on_image(self, img: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            assert hasattr(t, 'on_image'), f'{t.__class__.__name__}.on_image has not been implemented.'
            img = t.on_image(img)
        return img

class ToTensor(object):
    def on_image(self, img: np.ndarray) -> np.ndarray:
        return np.asarray(img).astype(np.float32) / 255.

    def __call__(self, img, kpts, mask):
        return self.on_image(img), kpts, mask


class Normalize(object):

    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def on_image(self, img: np.ndarray) -> np.ndarray:
        img = img[:,:,:3]
        img -= self.mean
        img /= self.std
        if self.to_bgr:
            img = img.transpose(2, 0, 1).astype(np.float32)
        return img

    def __call__(self, img, kpts, mask):
        img = self.on_image(img)
        return img, kpts, mask


class ColorJitter(object):

    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, kpts, mask):
        image = np.asarray(self.color_jitter(Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, kpts, mask


class RandomBlur(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, kpts, mask):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)
        return image, kpts, mask


def make_transforms(is_train: bool=False):
    if is_train is True:
        transform = Compose(
            [
                RandomBlur(0.5),
                ColorJitter(0.1, 0.1, 0.05, 0.05),
                ToTensor(),
                Normalize(mean=pvnet_mean, std=pvnet_std),
            ]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=pvnet_mean, std=pvnet_std),
            ]
        )

    return transform
