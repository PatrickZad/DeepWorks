import random
import collections.abc as cabc
import numpy as np
import cv2


def randRescaleAndTranspose(scale, *originalArray):
    if isinstance(scale, cabc.Iterable):
        shortLen = scale[random.randint(0, len(scale) - 1)]
    else:
        shortLen = scale
    scaledImgs = []
    for img in originalArray:
        if isinstance(img, str):
            img = cv2.imread(img)
        # multi-scale and transpose
        shape = img.shape
        reshape = [0, 0]
        if shape[0] > shape[1]:
            reshape[0] = shape[0] * shortLen / shape[1]
            reshape[1] = shortLen
        else:
            reshape[1] = shape[1] * shortLen / shape[0]
            reshape[0] = shortLen
        scaledImg = cv2.resize(img, (int(reshape[1]), int(reshape[0]))).transpose((2, 0, 1))
        # scaledImg=scaledImg.astype(np.float64)
        scaledImgs.append(scaledImg)
    return scaledImgs


def randCrop(crop, *images):
    shape = images[0].shape
    h_limit = shape[1] - crop
    w_limit = shape[2] - crop
    h_offset = random.randint(0, h_limit)
    w_offset = random.randint(0, w_limit)
    result = []
    for img in images:
        croped = img[:, h_offset:h_offset + crop, w_offset:w_offset + crop]
        result.append(croped)
    return result


def randHFlip(*images):
    flip = random.randint(0, 1)
    if flip > 0:
        result = []
        for img in images:
            fliped = np.flip(img, 2)
            result.append(fliped)
        return result
    return images


def rgbAlter(image):
    alphas = np.random.normal(0, 0.1, 3)
    for i in range(224):
        for j in range(224):
            origin = image[:, i, j].squeeze()
            mean = origin / 3
            row = origin.reshape((1, 3)) - mean
            column = origin.reshape((3, 1)) - mean
            cov = np.matmul(column, row)  # covariance
            eigvals, eigvecs = np.linalg.eig(cov)  # eigen-decompose
            sortedvals = np.sort(eigvals)
            mat = eigvecs[eigvals.index(sortedvals[-1]), :]
            mat.reshape((1, 3))
            for k in range(2, 0, -1):
                mat = np.concatenate([mat, eigvecs[eigvals.index(sortedvals[k])].reshape(1, 3)])
            mat = mat.transpose((1, 0))
            vec = np.array([alphas[0] * sortedvals[2], alphas[1] * sortedvals[1], alphas[2] * sortedvals[0]]).reshape(
                (3, 1))
            product = np.matmul(mat, vec).squeeze()
            for k in range(0, 3):
                image[k][i][j] += product[k]
