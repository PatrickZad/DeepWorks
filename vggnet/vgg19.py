import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import local_dataset

class VGG19(nn.Module):

    # pool is "max" or "average"
    def __init__(self, classNum, pool="max"):
        self.conv1 = conv1(pool)
        self.conv2 = conv2(pool)
        self.conv3 = conv3(pool)
        self.conv4 = conv4(pool)
        self.conv5 = conv5(pool)
        self.fullCon = fullConnectedSoftm(classNum)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        return self.fullCon(out5)


def poolSelect(pool):
    if pool is "average":
        return nn.AvgPool2d(kernel_size=2, stride=2)
    return nn.MaxPool2d(kernel_size=2, stride=2)


def conv1(pool="max"):
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(True),
        poolSelect(pool)
    )


def conv2(pool="max"):
    return nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(True),
        poolSelect(pool)
    )


def conv3(pool="max"):
    return nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        poolSelect(pool)
    )


def conv4(pool="max"):
    return nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        poolSelect(pool)
    )


def conv5(pool="max"):
    return nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(True),
        poolSelect(pool)
    )


def fullConnectedSoftm(classNum):
    return nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, classNum),
        nn.Softmax()
    )


def loadModel():
    pass


def trainModel(vgg, dataLoader, loss, optim, repeat=5000):
    pass


def testModel(vgg, testDataset):
    pass


if __name__ == "__main__":
    trainpath = ""
    testpath = ""
    vgg = VGG19()
    trainset = local_dataset.Cifar10Train()
    dataLoader = DataLoader(trainset, batch_size=256)
    loss = nn.CrossEntropyLoss()
    optim = optim.SGD()

