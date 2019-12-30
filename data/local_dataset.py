from torch.utils.data.dataset import Dataset
import torch
import os
import re
import pickle
import numpy as np
import sys
import re
import skimage.io as imgio
from skimage.transform import resize
import random
import data.augmentation as aug
import cv2

if re.match(r'.*inux.*', sys.platform):
    imagenetdir = r'/run/media/patrick/6114f130-f537-4999-b5f6-33fe2afc51db/imagenet12'
    cifar10dir = r'/run/media/patrick/6114f130-f537-4999-b5f6-33fe2afc51db/cifar10'
    facadeDir = r'/home/patrick/PatrickWorkspace/datasets/facades'
    cityscapesDir = r'/home/patrick/PatrickWorkspace/Datasets/cityscapes'
    edges2shoesDir = r'/home/patrick/PatrickWorkspace/Datasets/edges2shoes'
else:
    imagenetdir = r''
    cifar10dir = r''
    facadeDir = r''
    cityscapesDir = r''
    edges2shoesDir = r''
imagenetSubpath = {'img': {'train': 'img_train', 'test': 'img_test', 'train_t3': 'img_train_t3', 'val': 'img_val'}, \
                   'devkit': {'t3': 'ILSVRC2012_devkit_t3', 't12': 'ILSVRC2012_devkit_t12'},
                   'bbox': {'test_gogs': 'bbox_test_gogs', \
                            'train_dogs': 'bbox_train_dogs', 'train_v2': 'bbox_train_v2', 'val_v3': 'bbox_val_v3'}}
cifar10Subpath = {'train': [], 'test': []}


class Cifar10Train(Dataset):
    def __init__(self):
        parent = cifar10dir
        self.imageTensorList = []
        self.lableTensorList = []
        for batchFile in cifar10Subpath['train']:
            with open(os.path.join(parent, batchFile), 'rb') as file:
                batchDict = pickle.load(file, encoding='bytes')
                for i in range(len(batchDict[b'data'])):
                    imageArray = np.array(batchDict[b'data'][i]).reshape((3, 32, 32))
                    imageTensor = torch.from_numpy(imageArray).float()
                    self.imageTensorList.append(imageTensor)
                    lableTensor = torch.zeros(10).float()
                    lableTensor[batchDict[b'labels'][i]] = 1
                    self.lableTensorList.append(lableTensor)

    def __len__(self):
        return len(self.imageTensorList)

    def __getitem__(self, item):
        return self.imageTensorList[item], self.lableTensorList[item]


class Cifar10Test(Dataset):
    def __init__(self):
        self.imageTensorList = []
        self.lableTensorList = []
        for batchFile in cifar10Subpath['test']:
            with open(os.path.join(cifar10dir, batchFile), 'rb') as file:
                batchDict = pickle.load(file, encoding='bytes')
                for i in range(len(batchDict[b'data'])):
                    imageArray = np.array(batchDict[b'data'][i]).reshape((3, 32, 32))
                    imageTensor = torch.from_numpy(imageArray).float()
                    self.imageTensorList.append(imageTensor)
                    lableTensor = torch.zeros(10).float()
                    lableTensor[batchDict[b'labels'][i]] = 1
                    self.lableTensorList.append(lableTensor)

    def __len__(self):
        return len(self.imageTensorList)

    def __getitem__(self, item):
        return self.imageTensorList[item], self.lableTensorList[item]


class ImagenetTrain(Dataset):
    def __init__(self):
        from scipy.io import loadmat
        self.ID_lable_dict = {}
        self.ID_amount_dict = {}
        devkitdir = os.path.join(imagenetdir, imagenetSubpath['devkit']['t2'])
        devkitdir = os.path.join(devkitdir, 'data')
        devkitmeta = os.path.join(devkitdir, 'meta.mat')
        mat = loadmat(devkitmeta)
        for info in mat['synsets']:
            id = info[0][0][1][0]
            lable = info[0][0][0][0][0]
            self.ID_lable_dict[id] = lable
        traindir = os.path.join(imagenetdir, imagenetSubpath['img']['train'])
        self.dataLength = 0;
        for id in os.listdir(traindir):
            dir = os.path.join(traindir, id)
            length = len(os.listdir(dir))
            self.ID_amount_dict[id] = length
            self.dataLength += length

    def __len__(self):
        return self.dataLength

    def __getitem__(self, item):
        objid = None
        sum = -1
        for id, amount in self.ID_amount_dict.items():
            if item > sum and item <= sum + amount:
                objid = id
                break
            else:
                sum += amount
        out = torch.zeros(1000)
        out[self.ID_lable_dict[objid]] = 1
        dir = os.path.join(imagenetdir, imagenetSubpath['img']['train'])
        dir = os.path.join(dir, objid)
        filename = os.listdir(dir)[item - sum]
        originImg = imgio.imread(os.path.join(dir, filename))
        input = dataAugment(originImg)
        return input, out


def dataAugment(originalImgArray, scale=range(256, 513)):
    # cast to float
    originalImgArray = originalImgArray.astype(np.float32)
    # preprocesse to make it centered
    mean = np.sum(originalImgArray) / originalImgArray.size
    originalImgArray -= mean
    # multi-scale and reshape
    shortLen = scale[random.randint(0, len(scale) - 1)]
    shape = originalImgArray.shape
    if shape[0] > shape[1]:
        shape[0] *= shortLen / shape[1]
        shape[1] = shortLen
    else:
        shape[1] *= shortLen / shape[0]
        shape[0] = shortLen
    scaledImg = resize(originalImgArray, shape).transpose((2, 0, 1))
    # randome crop
    shape = scaledImg.shape
    h_limt = shape[0] - 224
    w_limt = shape[1] - 224
    h_offset = random.randint(0, h_limt)
    w_offset = random.randint(0, w_limt)
    cropped = scaledImg[:, h_offset:h_offset + 224, w_offset:w_offset + 224]
    # horizontal reflection/flip
    rand = random.randint(0, 1)
    if rand > 0:
        cropped = np.flip(cropped, 2)
    # alter RGB intensities
    alphas = np.random.normal(0, 0.1, 3)
    for i in range(224):
        for j in range(224):
            origin = cropped[:, i, j].squeeze()
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
                cropped[k][i][j] += product[k]
    # return ndarray
    return cropped


class ImagenetTest(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class ImagenetVal(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class GenerativeBasic(Dataset):
    def __init__(self):
        self.dir = None
        self.files = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        imagepair = cv2.imread(os.path.join(self.dir, self.files[item]))
        real, label = np.split(imagepair, 2, axis=1)
        results = aug.randRescaleAndTranspose(286, real, label)
        results = aug.randCrop(256, *results)
        results = aug.randHFlip(*results)
        return results[0].copy(), results[1].copy()


class FacadesTrain(GenerativeBasic):
    def __init__(self):
        self.dir = os.path.join(facadeDir, 'train')
        self.files = os.listdir(self.dir)


class FacadesTest(GenerativeBasic):
    def __init__(self):
        self.dir = os.path.join(facadeDir, 'test')
        self.files = os.listdir(self.dir)


class FacadesVal(GenerativeBasic):
    def __init__(self):
        self.dir = os.path.join(facadeDir, 'val')
        self.files = os.listdir(self.dir)


class CityscapesTrain(GenerativeBasic):
    def __init__(self):
        self.dir = os.path.join(cityscapesDir, 'train')
        self.files = os.listdir(self.dir)


class CityscapesVal(GenerativeBasic):
    def __init__(self):
        self.dir = os.path.join(cityscapesDir, 'val')
        self.files = os.listdir(self.dir)


class Edges2shoesTrain(GenerativeBasic):
    def __init__(self):
        self.dir = os.path.join(edges2shoesDir, 'train')
        self.files = os.listdir(self.dir)


class Edges2shoesVal(GenerativeBasic):
    def __init__(self):
        self.dir = os.path.join(edges2shoesDir, 'val')
        self.files = os.listdir(self.dir)
