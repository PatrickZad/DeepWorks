import skimage.io as imgio
from skimage.transform import resize
import random


def imageAugmentation(image, scale, flip=True):
    if isinstance(image, str):
        imageArray = imgio.imread(image)
        image = imageArray.transpose((2, 0, 1))
    # TODO cast to float
    # TODO multi-scale and reshape
    # TODO randome crop
    # TODO horizontal reflection/flip
    # TODO alter RGB intensities
