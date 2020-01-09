import torch
import torch.nn as nn
from model.pix2pix.pix2pix import PatchDiscriminator70
from model import NNconfig, batch_norm, inst_norm
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import os
'''config'''


class ModelConfig(NNconfig):
    def __init__(self, experiments_config):
        self.lr = 2e-4
        self.momentum_beta1 = 0.5
        self.momentum_beta2 = 0.999
        self.norm = inst_norm
        self.optim_coefficient_d = 0.5
        self.optim_coefficient_g = 1
        self.epoch = 200
        self.cuda = torch.cuda.is_available()
        self.save_dir = experiments_config.out_base
        self.log_dir = experiments_config.log_base
        # may need to change
        self.in_channel = 3
        self.out_channel = 3
        self.model_name = None
        self.conditional = True
        self.batch_size = 1
        self.loss_coefficient_forward = 10
        self.loss_coefficient_backward = 10
        self.print_loss = True
        self.lr_steady_epoch = 100
        self.lr_decay = 100
        # only for cycle gan
        self.buffer_size = 50

    def optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.lr, betas=(self.momentum_beta1, self.momentum_beta2))

    def main_loss(self):
        return nn.MSELoss()

    def schedule_lamda(self):
        def sch_func(epoch):
            lr_decay = 1.0 - max(0, epoch - self.lr_steady_epoch) / float(self.lr_decay + 1)
            return lr_decay

        return sch_func


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
        self.sequence = random.shuffle(range(length))
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
        optim_gx = self.config.optimizaer(self.generator_x.parameters())
        scheduler_gx = LambdaLR(optim_gx, self.config.schedule_lamda())
        optim_gy = self.config.optimizaer(self.generator_y.parameters())
        scheduler_gy = LambdaLR(optim_gy, self.config.schedule_lamda())
        optim_dx = self.config.optimizaer(self.discriminator_x.parameters())
        scheduler_dx = LambdaLR(optim_dx, self.config.schedule_lamda())
        optim_dy = self.config.optimizaer(self.discriminator_y.parameters())
        scheduler_dy = LambdaLR(optim_dy, self.config.schedule_lamda())
        rec_lossfunc_gx = torch.nn.L1Loss()
        rec_lossfunc_gy = torch.nn.L1Loss()
        gan_lossfunc_gx = self.config.main_loss()
        gan_lossfunc_gy = self.config.main_loss()
        gan_lossfunc_dx_g = self.config.main_loss()
        gan_lossfunc_dx_r = self.config.main_loss()
        gan_lossfunc_dy_g = self.config.main_loss()
        gan_lossfunc_dy_r = self.config.main_loss()
        for epoch in range(self.config.epoch):
            for step, (data_x, data_y) in enumerate(zip(dataloader_x, dataloader_y)):
                # generator_x,x to y direction
                optim_gx.zere_grad()
                gen_x = self.generator_x(data_x)
                dis_x = self.discriminator_x(gen_x)
                rec_x = self.generator_y(gen_x)
                dis_x_target = torch.ones(data_x.shape[0])
                gan_loss_gx = gan_lossfunc_gx(dis_x, dis_x_target)
                rec_loss_gx = rec_lossfunc_gx(data_x, rec_x)
                loss_gx = gan_loss_gx + self.config.loss_coefficient_forward * rec_loss_gx
                loss_gx.bacward()
                optim_gx.step()
                generated_buffer_x.add_new(gen_x)
                # generator_y,y to x direction
                optim_gy.zere_grad()
                gen_y = self.generator_y(data_y)
                dis_y = self.discriminator_y(gen_y)
                rec_y = self.generator_x(gen_y)
                dis_y_target = torch.ones(data_y.shape[0])
                gan_loss_gy = gan_lossfunc_gy(dis_y, dis_y_target)
                rec_loss_gy = rec_lossfunc_gy(data_y, rec_y)
                loss_gy = gan_loss_gy + self.config.loss_coefficient_backward * rec_loss_gy
                loss_gy.bacward()
                optim_gy.step()
                generated_buffer_y.add_new(gen_y)
                # discriminator x
                optim_dx.zero_grad()
                gen_x = generated_buffer_x.next_batch()
                dx_gen = self.discriminator_x(gen_x)
                dx_real = self.discriminator_x(data_y)
                dx_gen_target = torch.zeros(gen_x.shape[0])
                dx_real_target = torch.ones(data_y.shape[0])
                dx_loss = 0.5 * (gan_lossfunc_dx_g(dx_gen, dx_gen_target) +
                                 gan_lossfunc_dx_r(dx_real, dx_real_target))
                dx_loss.backward()
                optim_dx.step()
                # discriminator y
                optim_dy.zero_grad()
                gen_y = generated_buffer_y.next_batch()
                dy_gen = self.discriminator_y(gen_y)
                dy_real = self.discriminator_y(data_x)
                dy_gen_target = torch.zeros(gen_y.shape[0])
                dy_real_target = torch.ones(data_x.shape[0])
                dy_loss = 0.5 * (gan_lossfunc_dy_g(dy_gen, dy_gen_target) +
                                 gan_lossfunc_dy_r(dy_real, dy_real_target))
                dy_loss.backward()
                optim_dy.step()
            # scheduler
            scheduler_dx.step()
            scheduler_dy.step()
            scheduler_gx.step()
            scheduler_gy.step()

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
        info = '_' + info + '_'
        torch.save(self.generator_x.state_dict(),
                   os.path.join(self.config.save_dir, self.config.model_name + info + 'generator_x.pt'))
        torch.save(self.discriminator_x.state_dict(),
                   os.path.join(self.config.save_dir, self.config.model_name + info + 'discriminator_x.pt'))
        torch.save(self.generator_y.state_dict(),
                   os.path.join(self.config.save_dir, self.config.model_name + info + 'generator_y.pt'))
        torch.save(self.discriminator_y.state_dict(),
                   os.path.join(self.config.save_dir, self.config.model_name + info + 'discriminator_y.pt'))
