import vggnet.vgg19 as vggnet
import data.loclocal_dataset
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import skimage.io
import skimage.transform


class StyleTransfer:
    def __init__(self):
        self.vgg = self.__buildAverageVgg()

    def __init__(self, vgg):
        self.vgg = vgg

    def __buildAverageVgg(self):
        if os.path.exists(r'./averageVgg.pkl'):
            with open(r'./averageVgg.pkl', 'rb') as vggfile:
                model = pickle.load(vggfile, encoding='bytes')
            return model
        vgg = vggnet.VGG19(10, 'average')
        trainset = local_dataset.Cifar10Train()
        validset = local_dataset.Cifar10Test()
        vggnet.trainModel(vgg, trainset, validset)
        testset = local_dataset.Cifar10Test()
        print("test accuracy:" + vggnet.testModel(vgg, testset))
        with open(r'./averageVgg.pkl', 'wb') as modelFile:
            pickle.dump(vgg, modelFile)
        return vgg

    def contentloss(self, content, currentGen, layer='conv4'):
        contentConv = self.vgg.conv1(content)
        genConv = self.vgg.conv1(currentGen)
        if layer is 'conv2':
            contentConv = self.vgg.conv1(contentConv)
            genConv = self.vgg.conv1(genConv)
        if layer is 'conv3':
            for i in range(2):
                contentConv = self.vgg.conv1(contentConv)
                genConv = self.vgg.conv1(genConv)
        if layer is 'conv4':
            for i in range(3):
                contentConv = self.vgg.conv1(contentConv)
                genConv = self.vgg.conv1(genConv)
        lossFunc = nn.MSELoss(size_average=False)
        return 0.5 * lossFunc(contentConv, genConv)

    def styleloss(self, style, currentGen, weights=[0.2 for i in range(5)]):
        index = 0;
        styleConv = self.vgg.conv1(style)
        genConv = self.vgg.conv1(currentGen)
        size = styleConv.size()
        stylemat = styleConv.view(size[0], size[1] * size[2])
        genmat = genConv.view(size[0], size[1] * size[2])
        styleGram = torch.mm(stylemat, stylemat.t())
        genGram = torch.mm(genmat, genmat.t())
        lossFunc = nn.MSELoss()
        loss = weights[index] * 0.25 * lossFunc(styleGram, genGram)
        return loss

    def totalloss(self, content, style, currentGen, weights=[10 ** (-3), 1]):
        return weights[0] * self.contentloss(content, currentGen) + weights[1] * self.styleloss(style, currentGen)

    def whitenoise(self, size):
        noiseTensor = torch.from_numpy(np.random.standard_normal(size))
        noiseTensor.requires_grad = True
        return noiseTensor

    def transfer(self, contentTensor, styleTensor, repeat=5000, minchange=10 ** (-5)):
        currentGen = self.whitenoise(contentTensor.size())
        optimizer = torch.optim.Adam(currentGen, lr=10 ** (-2))
        lastloss = 0
        for epoch in range(repeat):
            loss = self.totalloss(contentTensor, styleTensor, currentGen)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print loss
            if epoch % 50 == 0:
                print('total loss=', loss.data[0])
            if loss.data[0] - lastloss < minchange:
                return currentGen
            lastloss = loss.data[0]
        return currentGen

    def styleTrans(self, contentImagePath, styleImagePath):
        contentArray = skimage.io.imread(contentImagePath)
        styleArray = skimage.io.imread(styleImagePath)
        contentSize = contentArray.shape
        styleSize = styleArray.shape
        height = min(contentSize[0], styleSize[0])
        width = min(contentSize[1], styleSize[1])
        contentArray = skimage.transform.resize(contentArray, (height, width, 3))
        styleArray = skimage.transform.resize(styleArray, (height, width, 3))
        contentTensor = torch.from_numpy(contentArray)
        contentTensor = torch.transpose(contentTensor, 1, 2)
        contentTensor = torch.transpose(contentTensor, 0, 1)
        styleTensor = torch.from_numpy(styleArray)
        styleTensor = torch.transpose(styleTensor, 1, 2)
        styleTensor = torch.transpose(styleTensor, 0, 1)
        result = self.transfer(contentTensor, styleTensor)
        result.transpose(0, 1)
        result.transpose(1, 2)
        return result.numpy()


if __name__ == '__main__':
    from torchvision.models import vgg19
    import imageio

    vgg = vgg19(True)
    transfer = StyleTransfer(vgg)
    result = transfer.styleTrans(r'../data/content.jpg', r'../data/content.jpg')
    imageio.imwrite(r'../data/trans.jpg', result)
