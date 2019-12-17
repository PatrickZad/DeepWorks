import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import data.local_dataset
import pickle

configs = {
    'A': [],
    'B': [],
    'C': [],
    'D': [],
    'E': []
}


class VGGnet(nn.Module):

    # pool is "max" or "average"
    def __init__(self, classNum, pool="max"):
        super(VGGnet, self).__init__()
        self.conv1 = self.__conv1(pool)
        self.conv2 = self.__conv2(pool)
        self.conv3 = self.__conv3(pool)
        self.conv4 = self.__conv4(pool)
        self.conv5 = self.__conv5(pool)
        self.fullCon = self.__fullConnectedSoftm(classNum)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        return self.fullCon(out5)

    def __poolSelect(self, pool):
        if pool is "average":
            return nn.AvgPool2d(kernel_size=2, stride=2)
        return nn.MaxPool2d(kernel_size=2, stride=2)

    def __conv1(self, pool="max"):
        conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(conv1_1.weight)
        conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(conv1_2.weight)
        return nn.Sequential(
            conv1_1,
            nn.ReLU(True),
            conv1_2,
            nn.ReLU(True),
            self.__poolSelect(pool)
        )

    def __conv2(self, pool="max"):
        conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(conv2_1.weight)
        conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(conv2_2.weight)
        return nn.Sequential(
            conv2_1,
            nn.ReLU(True),
            conv2_2,
            nn.ReLU(True),
            self.__poolSelect(pool)
        )

    def __conv3(self, pool="max"):
        conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(conv3_1.weight)
        conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(conv3_2.weight)
        conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(conv3_3.weight)
        conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(conv3_4.weight)
        return nn.Sequential(
            conv3_1,
            nn.ReLU(True),
            conv3_2,
            nn.ReLU(True),
            conv3_3,
            nn.ReLU(True),
            conv3_4,
            nn.ReLU(True),
            self.__poolSelect(pool)
        )

    def __conv4(self, pool="max"):
        conv4model = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            self.__poolSelect(pool)
        )
        for layer in conv4model.children():
            if type(layer) is 'torch.nn.Conv2d':
                nn.init.xavier_uniform_(layer.weight)
        return conv4model

    def __conv5(self, pool="max"):
        conv5model = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            self.__poolSelect(pool)
        )
        for layer in conv5model.children():
            if type(layer) is 'torch.nn.Conv2d':
                nn.init.xavier_uniform_(layer.weight)
        return conv5model

    def __fullConnectedSoftm(self, classNum):
        conmodel = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, classNum),
            nn.Softmax()
        )
        for layer in conmodel.children():
            if type(layer) is 'torch.nn.Linear':
                nn.init.xavier_uniform_(layer.weight)
        return conmodel

    def trainModel(self, trainDataset, validDataset, repeat=5000, minchange=10 ** (-5)):
        learningRate = 10 ** (-2)
        optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=learningRate, weight_decay=5 * 10 ** (-4))
        lossFunc = nn.CrossEntropyLoss()
        trainLoader = DataLoader(trainDataset, batch_size=256)
        validLoader = DataLoader(validDataset, batch_size=256)
        lastLoss = 0
        lastAccuracy = 0
        for epoch in range(repeat):
            for step, (x, y) in enumerate(trainLoader):
                output = self(x)
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
                accuracy = self.__validate(validLoader)
                if abs(accuracy - lastAccuracy) < minchange:
                    learningRate /= 10
                    optimizer = optim.SGD(momentum=0.9, lr=learningRate)

    def __validate(self, validateLoader):
        accuracy = 0
        for image, lable in validateLoader:
            out = self(image)
            outMaxIndex = torch.max(out, 0)[1]
            lableMaxIndex = torch.max(lable, 0)[0]
            if outMaxIndex[0] == lableMaxIndex[0]:
                accuracy += 1
        return accuracy / 256

    def testModel(self, testDataset):
        # TODO need to calculate top-1 and top-5 error
        error = 0
        for image, lable in testDataset.imageTensorList, testDataset.lableTensorList:
            out = self(image)
            outMaxIndex = torch.max(out, 0)[1]
            lableMaxIndex = torch.max(lable, 0)[0]
            if outMaxIndex[0] != lableMaxIndex[0]:
                error += 1
        return 1 - error / len(testDataset.imageTensorList)

    def savemodel(self):
        with open(r'./vggmodel.pkl', 'wb') as modelFile:
            pickle.dump(self, modelFile)


def loadModel():
    with open(r'./vggmodel.pkl', 'rb') as modelFile:
        result = pickle.load(modelFile, encoding='bytes')
    return result


if __name__ == "__main__":
    pass