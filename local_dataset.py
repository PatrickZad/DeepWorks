from torch.utils.data.dataset import Dataset
import torch
import os
import re
import pickle
import numpy as np


class Cifar10Train(Dataset):
    def __init__(self):
        parent = r'./data/cifar-10-batches-py'
        pattern = r'data_batch.*'
        self.imageTensorList = []
        self.lableTensorList = []
        for batchFile in os.listdir(parent):
            if re.match(pattern, batchFile):
                with open(batchFile, 'rb') as file:
                    batchDict = pickle.load(file, encoding='bytes')
                    for i in range(len(batchDict['data'])):
                        imageArray = np.array(batchDict['data'][i]).reshape((3, 32, 32))
                        imageTensor = torch.from_numpy(imageArray)
                        self.imageTensorList.append(imageTensor)
                        lableTensor = torch.tensor(list(batchDict['lable'][i]))
                        self.lableTensorList.append(lableTensor)

    def __len__(self):
        return len(self.imageTensorList)

    def __getitem__(self, item):
        return self.imageTensorList[item], self.lableTensorList[item]
