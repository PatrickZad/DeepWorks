from torch.utils.data.dataset import Dataset
import torch
import os
import re
import pickle
import numpy as np
import sys
import re

if re.match(r'.*inux.*', sys.platform()):
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
        self.ID_lable_dict={}
        self.ID_amount_dict={}
        devkitdir=os.path.join(imagenetdir,imagenetSubpath['devkit']['t2'])
        devkitdir=os.path.join(devkitdir,'data')
        devkitmeta=os.path.join(devkitdir,'meta.mat')
        mat=loadmat(devkitmeta)
        for info in mat['synsets']:
            id=info[0][0][1][0]
            lable=info[0][0][0][0][0]
            self.ID_lable_dict[id]=lable
        traindir = os.path.join(imagenetdir, imagenetSubpath['img']['train'])
        self.dataLength=0;
        for id in os.listdir(traindir):
            dir=os.path.join(traindir,id)
            length=len(os.listdir(dir))
            self.ID_amount_dict[id]=length
            self.dataLength+=length

    def __len__(self):
        return self.dataLength

    def __getitem__(self, item):

        pass


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
