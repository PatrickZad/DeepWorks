from torch.utils.data.dataset import Dataset
import torch
import os
import re
import numpy as np
import sys
import re
import skimage.io as imgio
from skimage.transform import resize
import random
from data.manipulation import toFloat, randRescaleAndTranspose, randHFlip, randCrop, rgbAlter
import cv2
import pickle
from experiments import platform_linux, platform_kaggle, platform_win, platform_kaggle_test

data_bases = ['/home/patrick/PatrickWorkspace/Datasets', '/kaggle/input', '',
              '/home/patrick/PatrickWorkspace/Datasets/kaggle_test/input']

if re.match(r'.*inux.*', sys.platform):
    imagenetdir = r'/run/media/patrick/6114f130-f537-4999-b5f6-33fe2afc51db/imagenet12'
    cifar10dir = r'/run/media/patrick/6114f130-f537-4999-b5f6-33fe2afc51db/cifar10'
else:
    imagenetdir = r''
    cifar10dir = r''

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


class ImagePairBasic(Dataset):

    def read_dir(self, dir):
        files = os.listdir(dir)
        pairlist = []
        for file in files:
            image = cv2.imread(os.path.join(dir, file))
            image = randRescaleAndTranspose(286, image)
            pairlist.append(image)
        self.pairs_array = np.concatenate(pairlist, 0)

    def read_binary(self, binary):
        with open(binary, 'rb') as file:
            self.pairs_array = pickle.load(file)

    def __len__(self):
        return self.pairs_array.shape[0]

    def __getitem__(self, item):
        imagepair = self.pairs_array[item]
        results = np.split(imagepair, 2, axis=2)
        results = randCrop(256, *results)
        results = randHFlip(*results)
        float_imgs = toFloat(*results, symmetric=True)
        return float_imgs

    def toBinary(self, file):
        with open(file, 'wb') as pkl:
            pickle.dump(self.pairs_array, pkl)


class ImagePairVal(Dataset):
    def read_dir(self, dir):
        files = os.listdir(dir)
        pairlist = []
        for file in files:
            image = cv2.imread(os.path.join(dir, file))
            image = image.transpose((2, 0, 1))
            pairlist.append([image])
        self.pairs_array = np.concatenate(pairlist, 0)

    def read_binary(self, binary):
        with open(binary, 'rb') as file:
            self.pairs_array = pickle.load(file)

    def __len__(self):
        return self.pairs_array.shape[0]

    def __getitem__(self, item):
        imagepair = self.pairs_array[item]
        results = np.split(imagepair, 2, axis=2)
        float_imgs = toFloat(*results, symmetric=True)
        return float_imgs

    def toBinary(self, file):
        with open(file, 'wb') as pkl:
            pickle.dump(self.pairs_array, pkl)


class FacadesTrain(ImagePairBasic):

    def __init__(self, platform):
        if platform == platform_kaggle or platform == platform_kaggle_test:
            binary = os.path.join(data_bases[platform], 'facades_train.pkl')
            self.read_binary(binary)
        else:
            dir = os.path.join(data_bases[platform], 'facades', 'train')
            self.read_dir(dir)


class FacadesTest(ImagePairBasic):
    def __init__(self, platform):
        if platform == platform_kaggle or platform == platform_kaggle_test:
            binary = os.path.join(data_bases[platform], 'facades_test.pkl')
            self.read_binary(binary)
        else:
            dir = os.path.join(data_bases[platform], 'facades', 'test')
            self.read_dir(dir)


class FacadesVal(ImagePairVal):
    def __init__(self, platform):
        if platform == platform_kaggle or platform == platform_kaggle_test:
            binary = os.path.join(data_bases[platform], 'facades_val.pkl')
            self.read_binary(binary)
        else:
            dir = os.path.join(data_bases[platform], 'facades', 'val')
            self.read_dir(dir)


class CityscapesTrain(ImagePairBasic):
    def __init__(self, platform):
        if platform == platform_kaggle or platform == platform_kaggle_test:
            binary = os.path.join(data_bases[platform], 'cityscapes_train.pkl')
            self.read_binary(binary)
        else:
            dir = os.path.join(data_bases[platform], 'cityscapes', 'train')
            self.read_dir(dir)


class CityscapesVal(ImagePairVal):
    def __init__(self, platform):
        if platform == platform_kaggle or platform == platform_kaggle_test:
            binary = os.path.join(data_bases[platform], 'cityscapes_val.pkl')
            self.read_binary(binary)
        else:
            dir = os.path.join(data_bases[platform], 'cityscapes', 'val')
            self.read_dir(dir)


class Edges2shoesTrain(ImagePairBasic):
    def __init__(self, platform):
        if platform == platform_kaggle or platform == platform_kaggle_test:
            binary = os.path.join(data_bases[platform], 'edges2shoes_train.pkl')
            self.read_binary(binary)
        else:
            dir = os.path.join(data_bases[platform], 'edges2shoes', 'tarin')
            self.read_dir(dir)


class Edges2shoesVal(ImagePairVal):
    def __init__(self, platform):
        if platform == platform_kaggle or platform == platform_kaggle_test:
            binary = os.path.join(data_bases[platform], 'edges2shoes_val.pkl')
            self.read_binary(binary)
        else:
            dir = os.path.join(data_bases[platform], 'edges2shoes', 'val')
            self.read_dir(dir)


class MapTrain(ImagePairBasic):
    def __init__(self, platform):
        if platform == platform_kaggle or platform == platform_kaggle_test:
            binary = os.path.join(data_bases[platform], 'maps_train.pkl')
            self.read_binary(binary)
        else:
            dir = os.path.join(data_bases[platform], 'maps', 'train')
            self.read_dir(dir)


class MapVal(ImagePairVal):
    def __init__(self, platform):
        if platform == platform_kaggle or platform == platform_kaggle_test:
            binary = os.path.join(data_bases[platform], 'maps_val.pkl')
            self.read_binary(binary)
        else:
            dir = os.path.join(data_bases[platform], 'maps', 'val')
            self.read_dir(dir)


labels = [
    #   name    id      color
    ('unlabeled', 0, (0, 0, 0)),
    ('ego vehicle', 1, (0, 0, 0)),
    ('rectification border', 2, (0, 0, 0)),
    ('out of roi', 3, (0, 0, 0)),
    ('static', 4, (0, 0, 0)),
    ('dynamic', 5, (111, 74, 0)),
    ('ground', 6, (81, 0, 81)),
    ('road', 7, (128, 64, 128)),
    ('sidewalk', 8, (244, 35, 232)),
    ('parking', 9, (250, 170, 160)),
    ('rail track', 10, (230, 150, 140)),
    ('building', 11, (70, 70, 70)),
    ('wall', 12, (102, 102, 156)),
    ('fence', 13, (190, 153, 153)),
    ('guard rail', 14, (180, 165, 180)),
    ('bridge', 15, (150, 100, 100)),
    ('tunnel', 16, (150, 120, 90)),
    ('pole', 17, (153, 153, 153)),
    ('polegroup', 18, (153, 153, 153)),
    ('traffic light', 19, (250, 170, 30)),
    ('traffic sign', 20, (220, 220, 0)),
    ('vegetation', 21, (107, 142, 35)),
    ('terrain', 22, (152, 251, 152)),
    ('sky', 23, (70, 130, 180)),
    ('person', 24, (220, 20, 60)),
    ('rider', 25, (255, 0, 0)),
    ('car', 26, (0, 0, 142)),
    ('truck', 27, (0, 0, 70)),
    ('bus', 28, (0, 60, 100)),
    ('caravan', 29, (0, 0, 90)),
    ('trailer', 30, (0, 0, 110)),
    ('train', 31, (0, 80, 100)),
    ('motorcycle', 32, (0, 0, 230)),
    ('bicycle', 33, (119, 11, 32)),
    ('license plate', -1, (0, 0, 142))]

annos = ('color', 'instanceIds', 'labelIds')


class OriginCityscapes(Dataset):
    def __init__(self, dir, anno=annos[2], *city):
        self.anno_dir = os.path.join(dir, 'annotation')
        self.anno_postfix = '_000019_gtFine_'
        self.real_dir = os.path.join(dir, 'real')
        self.length = 0
        if city == 'all':
            self.cities = os.listdir(self.anno_dir)
        else:
            self.cities = (city,)
        self.ids = {}
        for city in self.cities:
            files = os.listdir(os.path.join(self.anno_dir, city))
            fileIds = [filename.split('_')[1] for filename in files]
            self.ids[city] = list(set(fileIds))
            self.length += len(self.ids[city])

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        #TODO
        less = None
        city = None
        offset=item
        for city,ids in self.ids.items():
            pass


class OriginCityscapesFineTrain(Dataset):
    pass


class OriginCityscapesFineVal(Dataset):
    pass


class OriginCityscapesFineTest(Dataset):
    pass


if __name__ == '__main__':
    maptrain = MapTrain(platform_linux)
    maptrain.toBinary(os.path.join(data_bases[platform_linux], 'maps', 'maps_train.pkl'))
