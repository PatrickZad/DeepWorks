import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from model import NNconfig
from torchvision.models import vgg16
from data.local_dataset import CityscapesTrain
from experiments import platform_linux

'''config'''


class ModelConfig(NNconfig):
    def __init__(self, experiments_config):
        self.lr = 1e-4
        self.momentum = 0.9
        self.optim_coefficient_d = 0.5
        self.optim_coefficient_g = 1
        self.epoch = 256
        self.cuda = torch.cuda.is_available()
        self.save_dir = experiments_config.out_base
        self.log_dir = experiments_config.log_base
        # may need to change
        self.in_channel = 3
        self.out_channel = None
        self.model_name = None
        self.batch_size = 20
        self.print_loss = True

    def optimizer(self, parameters):
        return torch.optim.SGD(parameters, lr=self.lr, momentum=self.momentum)

    def main_loss(self):
        return nn.BCELoss()


class FCN8s(nn.Module):
    def __init__(self):
        super(FCN8s, self).__init__()
        self.vgg = vgg16(pretrained=True)

    def forward(self, input):
        pass


def statistics():
    dataset = CityscapesTrain(platform_linux)
    dataloader = DataLoader(dataset, batch_size=1)
    labels = {}
    for real, label in dataloader:
        img = torch.squeeze(label).numpy()
        for i in range(256):
            for j in range(256):
                pixel = img[:, i, j]
                key = '('+str(pixel[0]) + ',' + str(pixel[1]) + ',' + str(pixel[2])+')'
                if key in labels.keys():
                    labels[key] += 1
                else:
                    labels[key] = 1
    print(labels)
if __name__=='__main__':
    statistics()