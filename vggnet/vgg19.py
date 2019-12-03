import torch.nn as nn
import torch.utils.data.dataset as Dataset

class VGG19(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass

class TrainData(Dataset):

    def __init__(self, file):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


def conv1Max():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def conv2Max():
    return nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def conv3Max():
    return nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def conv4Max():
    return nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def conv5Max():
    return nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def fullConnectedSoftm(classNum):
    return nn.Sequential(
        nn.Linear(512*7*7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, classNum),
        nn.Softmax()
    )
