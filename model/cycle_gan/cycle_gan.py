import torch
import torch.nn as nn
from model.pix2pix.pix2pix import PatchDiscriminator70
from model import NNconfig, batch_norm, inst_norm
import random
from torch.utils.data import DataLoader

'''config'''


class ModelConfig(NNconfig):
    def __init__(self, experiments_config):
        self.lr = 2e-4
        self.momentum_beta1 = 0.5
        self.momentum_beta2 = 0.999
        self.norm = inst_norm
        self.optim_coefficient_d = 0.5
        self.optim_coefficient_g = 1
        self.epoch = 256
        self.cuda = torch.cuda.is_available()
        self.save_dir = experiments_config.out_base
        self.log_dir = experiments_config.log_base
        # may need to change
        self.in_channel = 3
        self.out_channel = 3
        self.model_name = None
        self.conditional = True
        self.batch_size = 1
        self.loss_coefficient = 10
        self.print_loss = True
        # only for cycle gan
        self.buffer_size = 50

    def optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.lr, betas=(self.momentum_beta1, self.momentum_beta2))

    def change_optimizer(self, parameters, **kwargs):
        return torch.optim.Adam(parameters, kwargs)

    def main_loss(self):
        return nn.MSELoss()


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
        self.enc_conv = nn.Sequential(conv_layer_pad(config.in_channel, 64, kernel=7, stride=1, pad=3),
                                      conv_layer_pad(64, 128),
                                      conv_layer_pad(128, 256))
        self.res_layers = nn.Sequential(ResLayer())
        for i in range(8):
            self.res_layers.add_module(ResLayer())
        self.dec_conv = nn.Sequential(transpose_conv_layer_pad(256, 128),
                                      transpose_conv_layer_pad(128, 64),
                                      conv_layer_pad(64, config.out_channel, kernel=7, stride=1, pad=3),
                                      nn.Tanh())
        if config.cuda:
            self.enc_conv = self.enc_conv.cuda()
            self.res_layers = self.res_layers.cuda()
            self.dec_conv = self.dec_conv.cuda()

    def forward(self, x):
        enc = self.enc_conv(x)
        res = self.res_layers(enc)
        dec = self.dec_conv(res)
        return dec


class GeneratedBuffer:
    def __init__(self, samp_size, length):
        self.max_size = length
        self.actual_size = 0
        self.samp_size = samp_size
        self.sequence = random.shuffle(range(self.length))
        self.next = 0
        self.data = []

    def next_batch(self):
        samp = random.sample(range(self.actual_size), self.samp_size)
        batch = [torch.unsqueeze(self.data[i], 0) for i in samp]
        return torch.cat(batch, dim=0)

    def add_new(self, generate_batch):
        size = generate_batch.shape[0]
        if self.max_size - self.actual_size >= size:
            self.data += [generate_batch[i].copy() for i in range(size)]
        else:
            replace_size = size - (self.max_size - self.actual_size)
            replace_index = random.sample(range(self.actual_size), replace_size)
            for repl, src in zip(replace_index, range(replace_size)):
                self.data[repl] = generate_batch[src].copy()
            self.data += [generate_batch[i].copy() for i in range(replace_size, size)]


'''model'''


class CycleGAN:
    def __init__(self, config):
        self.config = config
        self.generator_x = BasicGenerator(config)
        self.generator_y = BasicGenerator(config)
        self.discriminator_x = PatchDiscriminator70(config)
        self.discriminator_y = PatchDiscriminator70(config)

    def train_model(self, dataset_x, dataset_y):
        generated_buffer_x = GeneratedBuffer(self.config.batch_size, self.config.buffer_size)
        generated_buffer_y = GeneratedBuffer(self.config.batch_size, self.config.buffer_size)
        dataloader_x = DataLoader(dataset_x, batch_size=self.config.batch_size, shuffle=True)
        dataloader_y = DataLoader(dataset_y, batch_size=self.config.batch_size, shuffle=True)
        for epoch in range(self.config.epoch):
            for step, (data_x, data_y) in enumerate(zip(dataloader_x, dataloader_y)):
                # generator_x

    def __call__(self, sketch_x=None, sketch_y=None):
        result = []
        if sketch_x is not None:
            generate_x = self.generator_x(sketch_x)
            result.append(generate_x)
        if sketch_y is not None:
            generate_y = self.generator_x(sketch_y)
            result.append(generate_y)
        return result

    def store(self, info):
        pass
