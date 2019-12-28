import skimage.io as imgio
from skimage.transform import resize
import random
import collections.abc as cabc
import numpy as np


def imageAugmentation(image, scale, crop, flip=True):
    # set scale
    if isinstance(scale, cabc.Iterable):
        shortLen = scale[random.randint(0, len(scale) - 1)]
    else:
        shortLen = scale
    # set to flip or not
    if flip:
        flip = random.randint(0, 1)
    else:
        flip = 0
    # do exactly same translations to a group of images
    if not isinstance(image, cabc.Iterable):
        image = (image)
    scaledImgs = []
    for img in image:
        if isinstance(img, str):
            img = imgio.imread(img)
            # cast to float
        img = img.astype(np.float32)
        # multi-scale and transpose
        shape = img.shape
        if shape[1] > shape[2]:
            shape[1] *= shortLen / shape[2]
            shape[2] = shortLen
        else:
            shape[2] *= shortLen / shape[1]
            shape[1] = shortLen
        scaledImg = resize(img, shape).transpose((2, 0, 1))
        scaledImgs.append(scaledImg)
    shape = scaledImgs[0].shape
    h_limit = shape[1] - crop
    w_limit = shape[2] - crop
    h_offset = random.randint(0, h_limit)
    w_offset = random.randint(0, w_limit)
    # randome crop,horizontal reflection/flip
    for scaled in scaledImgs:
        scaled = scaled[:, h_offset:h_offset + crop, w_offset:w_offset + crop]
        if flip > 0:
            scaled = np.flip(scaled, 2)
    return scaledImgs
