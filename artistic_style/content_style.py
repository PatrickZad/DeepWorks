import vggnet.vgg19 as vggnet
import local_dataset
import pickle
import os


class StyleTransfer:
    def __init__(self):
        self.vgg = self.__buildAverageVgg()

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
