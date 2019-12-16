from torch.utils.data.dataset import Dataset
import torch
import os
import re
import pickle
import numpy as np

absParent=r'/home/patrick/PatrickWorkspace/DeepWorks/data/cifar-10-batches-py'
absTestPath=r'/home/patrick/PatrickWorkspace/DeepWorks/data/cifar-10-batches-py/test_batch'
class Cifar10Train(Dataset):
    def __init__(self):
        parent = absParent
        pattern = r'data_batch.*'
        self.imageTensorList = []
        self.lableTensorList = []
        for batchFile in os.listdir(parent):
            if re.match(pattern, batchFile):
                with open(os.path.join(parent,batchFile), 'rb') as file:
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
        batchFile = absTestPath
        self.imageTensorList = []
        self.lableTensorList = []
        with open(batchFile, 'rb') as file:
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
