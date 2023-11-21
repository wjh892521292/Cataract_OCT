import cv2
import numpy as np
import numpy.random as random
from PIL import Image as Image, ImageEnhance
import torchvision.transforms as transforms

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            print(img.size)
            img = t(img)
        return img





class TrainAugmentation(object):
    def __init__(self, size=224, rescale=255., mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.size = size
        self.std = std
        self.rescale = rescale



        self.augment = Compose([
            transforms.Resize((self.size, self.size), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # elif mode == 'moco':
        #     self.augment = Compose([
        #         transforms.Resize((self.size, self.size), interpolation=2),
        #         transforms.RandomApply([
        #             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        #         ], p=0.8),
        #         transforms.RandomGrayscale(p=0.2),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.RandomVerticalFlip(p=0.5),
        #         transforms.RandomRotation(180),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
        # ])
        #
    def __call__(self, img):
        return self.augment(img)


class TestAugmentation(object):

    def __init__(self, size=224, rescale=255., mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.mean = mean
        self.size = size
        self.std = std
        self.rescale = rescale




        self.augment = Compose([
            transforms.Resize((self.size, self.size), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, img):
        return self.augment(img)



