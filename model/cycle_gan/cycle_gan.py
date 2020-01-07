import torch
import torch.nn as nn
from model.pix2pix.pix2pix import PatchDiscriminator70
from model import NNconfig, batch_norm, inst_norm

'''config'''


class ModelConfig(NNconfig):
    def __init__(self, experiments_config):
        pass

    def optimizer(self, parameters):
        pass

    def main_loss(self):
        pass


'''build model'''


def conv_layer_pad(in_channel, out_channel, kernel=3, stride=2, pad=1, relu=True):
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride)
    nn.init.normal_(conv.weight.data, 0, 0.02)
    norm = nn.InstanceNorm2d(out_channel)
    nn.init.normal_(norm.weight.data, 1, 0.02)
    layer = nn.Sequential(nn.ReflectionPad2d(padding=pad), conv, norm)
    if relu:
        layer.add_module(nn.ReLU())
    return layer


def transpose_conv_layer_pad(in_channel, out_channel, kernel=3, stride=2):
    conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride)
    nn.init.normal_(conv.weight.data, 0, 0.02)
    norm = nn.InstanceNorm2d(out_channel)
    nn.init.normal_(norm.weight.data, 1, 0.02)
    return nn.Sequential(nn.ReflectionPad2d(padding=1), conv, norm, nn.ReLU())


class ResLayer(nn.Module):
    def __init__(self):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(conv_layer_pad(256, 256, stride=1),
                                    conv_layer_pad(256, 256, stride=1, relu=False))

    def forward(self, x):
        return self.linear(x) + x


class BasicGenerator(nn.Module):
    def __init__(self, config):
        super(BasicGenerator, self).__init__()
        self.enc_conv = nn.Sequential(conv_layer_pad(config.in_channel, 64, stride=1, pad=3),
                                      conv_layer_pad(64, 128),
                                      conv_layer_pad(128, 256))
        self.res_layers = nn.Sequential(ResLayer())
        for i in range(8):
            self.res_layers.add_module(ResLayer())
        self.dec_conv = nn.Sequential(transpose_conv_layer_pad(256, 128),
                                      transpose_conv_layer_pad(128, 64),
                                      conv_layer_pad(64, config.out_channel, stride=1, pad=3))
        if config.cuda:
            self.enc_conv = self.enc_conv.cuda()
            self.res_layers = self.res_layers.cuda()
            self.dec_conv = self.dec_conv.cuda()

    def forward(self, x):
        enc = self.enc_conv(x)
        res = self.res_layers(enc)
        dec = self.dec_conv(res)
        return dec


class CycleGenerator(nn.Module):
    def __init__(self):
        super(CycleGenerator, self).__init__()
        pass

    def forward(self, x):
        pass


'''model'''


class CycleGAN:
    def __init__(self, config):
        pass

    def train_model(self, dataset):
        pass

    def __call__(self, sketch):
        pass

    def store(self, info):
        pass
