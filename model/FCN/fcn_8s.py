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
        self.out_labels = None

    def optimizer(self, parameters):
        return torch.optim.SGD(parameters, lr=self.lr, momentum=self.momentum)

    def main_loss(self):
        return nn.BCELoss()


class FCN8s(nn.Module):
    def __init__(self, config):
        super(FCN8s, self).__init__()
        self.config = config
        self.vgg = vgg16(pretrained=True)
        self.subnet1 = None  # conv1-pool3
        self.pool3_out = nn.Conv2d(256, self.config.out_labels, 1)
        self.subnet2 = None  # conv4-pool4
        self.pool4_out = nn.Conv2d(512, self.config.out_labels, 1)
        self.subnet3 = None  # conv5-conv7out
        self.pool4_upsample = None
        self.upsamp2_7 = nn.ConvTranspose2d(self.config.out_labels, self.config.out_labels, kernel_size=4, stride=2,
                                            padding=1, bias=False)

        self.upsamp2_4_7 = nn.ConvTranspose2d(self.config.out_labels, self.config.out_labels, kernel_size=4, stride=2,
                                              padding=1, bias=False)
        self.upsamp8_final = nn.ConvTranspose2d(self.config.out_labels, self.config.out_labels, kernel_size=16,
                                                stride=8, padding=4, bias=False)

    def forward(self, input):
        sub3_out = self.subnet1(input)
        sub3_scores = self.pool3_out(sub3_out)
        sub4_out = self.subnet2(sub3_out)
        sub4_scores = self.pool4_out(sub4_out)
        conv7_out = self.subnet3(sub4_out)
        conv_7by2 = self.upsamp2_7(conv7_out)
        upsamp_4_7 = self.upsamp2_4_7(conv_7by2 + sub4_scores)
        return self.upsamp8_final(upsamp_4_7 + sub3_scores)

    def train_model(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == '__main__':
    pass