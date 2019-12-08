import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import local_dataset
import pickle


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
        nn.Dropout(0.5),
        nn.Linear(4096, classNum),
        nn.Softmax()
    )


def loadModel():
    with open(r'./vggmodel.pkl', 'rb') as modelFile:
        result = pickle.load(modelFile, encoding='bytes')
    return result


def trainModel(vgg, trainDataset, validDataset, repeat=5000, minchange=10 ** (-5)):
    learningRate = 10 ** (-2)
    optimizer = optim.SGD(vgg.parameters(), momentum=0.9, lr=learningRate)
    lossFunc = nn.CrossEntropyLoss()
    trainLoader = DataLoader(trainset, batch_size=256)
    validLoader = DataLoader(validDataset, batch_size=256)
    lastLoss = 0
    lastAccuracy = 0
    for epoch in range(repeat):
        for step, (x, y) in enumerate(trainLoader):
            output = vgg(x)
            loss = lossFunc(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print loss
            if step % 10 == 0:
                print('loss:', loss.data[0])
            # stop if necessary
            if abs(loss.data[0] - lastLoss) < minchange:
                return
            lastLoss = loss.data[0]
            # validate and shrink learning rate if necessary
            accuracy = validset(vgg, validLoader)
            if abs(accuracy - lastAccuracy) < minchange:
                learningRate /= 10
                optimizer = optim.SGD(momentum=0.9, lr=learningRate)


def validate(vgg, validateLoader):
    accuracy = 0
    for image, lable in validateLoader:
        out = vgg(image)
        outMaxIndex = torch.max(out, 0)[1]
        lableMaxIndex = torch.max(lable, 0)[0]
        if outMaxIndex[0] == lableMaxIndex[0]:
            accuracy += 1
    return accuracy / 256


def testModel(vgg, testDataset):
    error = 0
    for image, lable in testDataset.imageTensorList, testDataset.lableTensorList:
        out = vgg(image)
        outMaxIndex = torch.max(out, 0)[1]
        lableMaxIndex = torch.max(lable, 0)[0]
        if outMaxIndex[0] != lableMaxIndex[0]:
            error += 1
    return 1 - error / len(testDataset.imageTensorList)


if __name__ == "__main__":
    vgg = VGG19(10)
    trainset = local_dataset.Cifar10Train()
    validset = local_dataset.Cifar10Test()
    trainModel(vgg, trainset, validset)
    testset = local_dataset.Cifar10Test()
    print("test accuracy:" + testModel(vgg, testset))
    with open(r'./vggmodel.pkl', 'wb') as modelFile:
        pickle.dump(vgg, modelFile)
