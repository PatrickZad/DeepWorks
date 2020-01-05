import torch
import torch.nn as nn
from collections import OrderedDict
import os
import logging
from model import NNconfig

'''config'''
# constant
batch_norm = 0
inst_norm = 1
generator_unet = 0
generator_basic = 1
discriminator_patch = 0
discriminator_pixel = 1
discriminator_image = 2


class ModelConfig(NNconfig):
    def __init__(self, experiments_config):
        self.lr = 2 * 10 ** (-4)
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
        self.l1_coeficient = 100
        self.print_loss = True

    def optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.lr, betas=(self.momentum_beta1, self.momentum_beta2))

    def main_loss(self):
        return nn.BCELoss()


'''build model'''


def norm_layer(features, norm=batch_norm):
    if norm == batch_norm:
        layer = nn.BatchNorm2d(features, track_running_stats=False)
    elif norm == inst_norm:
        layer = nn.InstanceNorm2d(features, affine=True)
    return layer


def conv_layer(in_channel, out_channel, kernel, stride, pad):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size=kernel, padding=pad, stride=stride)
    nn.init.normal_(conv.weight.data, mean=0, std=0.02)
    return conv


def conv_transpose_layer(in_channel, out_channel, kernel, stride, pad):
    conv = nn.ConvTranspose2d(
        in_channel, out_channel, kernel_size=kernel, padding=pad, stride=stride)
    nn.init.normal_(conv.weight.data, mean=0, std=0.02)
    return conv


def ck_layer(in_channel, out_channel, norm, kernel=4, stride=2, pad=1):
    if norm is None:
        layer = nn.Sequential(
            conv_layer(in_channel, out_channel, kernel, stride, pad),
            nn.LeakyReLU(0.2))
    else:
        layer = nn.Sequential(
            conv_layer(in_channel, out_channel, kernel, stride, pad),
            norm_layer(out_channel, norm),
            nn.LeakyReLU(0.2))
    return layer


def transpose_ck_layer(in_channel, out_channel, norm, kernel=4, stride=2, pad=1):
    if norm is None:
        layer = nn.Sequential(
            conv_transpose_layer(in_channel, out_channel, kernel, stride, pad),
            nn.ReLU())
    else:
        layer = nn.Sequential(
            conv_transpose_layer(in_channel, out_channel, kernel, stride, pad),
            norm_layer(out_channel, norm),
            nn.ReLU())
    return layer


def cdk_layer(in_channel, out_channel, norm, kernel=4, stride=2, pad=1):
    if norm is None:
        layer = nn.Sequential(
            conv_layer(in_channel, out_channel, kernel, stride, pad),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2))
    else:
        layer = nn.Sequential(
            conv_layer(in_channel, out_channel, kernel, stride, pad),
            norm_layer(out_channel, norm),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2))
    return layer


def transpose_cdk_layer(in_channel, out_channel, norm, kernel=4, stride=2, pad=1):
    if norm is None:
        layer = nn.Sequential(
            conv_transpose_layer(in_channel, out_channel, kernel, stride, pad),
            nn.Dropout2d(0.5),
            nn.ReLU())
    else:
        layer = nn.Sequential(
            conv_transpose_layer(in_channel, out_channel, kernel, stride, pad),
            norm_layer(out_channel, norm),
            nn.Dropout2d(0.5),
            nn.ReLU())
    return layer


def basic_encoder(norm, in_channels=3):
    return nn.Sequential(
        OrderedDict([('enc1', ck_layer(in_channels, 64, norm=None)),
                     ('enc2', ck_layer(64, 128, norm)),
                     ('enc3', ck_layer(128, 256, norm)),
                     ('enc4', ck_layer(256, 512, norm)),
                     ('enc5', ck_layer(512, 512, norm)),
                     ('enc6', ck_layer(512, 512, norm)),
                     ('enc7', ck_layer(512, 512, norm)),
                     ('enc8', ck_layer(512, 512, norm))]))


def basic_decoder(norm, out_channels=3):
    return nn.Sequential(transpose_cdk_layer(512, 512, norm),
                         transpose_cdk_layer(512, 512, norm),
                         transpose_cdk_layer(512, 512, norm),
                         transpose_ck_layer(512, 512, norm),
                         transpose_ck_layer(512, 256, norm),
                         transpose_ck_layer(256, 128, norm),
                         transpose_ck_layer(128, 64, norm),
                         nn.Sequential(nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
                                       nn.Tanh()))


def unet_decoder(norm, out_channels=3):
    return nn.Sequential(
        OrderedDict([('dec-7', transpose_cdk_layer(512, 512, norm)),
                     ('dec-6', transpose_cdk_layer(1024, 512, norm)),
                     ('dec-5', transpose_cdk_layer(1024, 512, norm)),
                     ('dec-4', transpose_cdk_layer(1024, 512, norm)),
                     ('dec-3', transpose_cdk_layer(1024, 256, norm)),
                     ('dec-2', transpose_cdk_layer(512, 128, norm)),
                     ('dec-1', transpose_cdk_layer(256, 64, norm)),
                     ('decout',
                      nn.Sequential(nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh()))]))


class BasicGenerator(nn.Module):
    def __init__(self, config):
        super(BasicGenerator, self).__init__()
        self.basic_encoder = basic_encoder(config.norm, config.in_channel)
        self.basic_decoder = basic_decoder(config.norm, config.out_channel)
        if config.cuda:
            self.basic_encoder = self.basic_encoder.cuda()
            self.basic_decoder = self.basic_decoder.cuda()

    def forward(self, sketch):
        encode = self.basic_encoder.encoder(sketch)
        decode = self.basic_decoder.decoder(encode)
        return self.basic_decoder.out(decode)


class UnetGenerator(nn.Module):
    def __init__(self, config):
        super(UnetGenerator, self).__init__()
        self.basic_encoder = basic_encoder(config.norm, config.in_channel)
        self.unet_decoder = unet_decoder(config.norm, config.out_channel)
        self.cuda = config.cuda
        if config.cuda:
            self.basic_encoder = self.basic_encoder.cuda()
            self.unet_decoder = self.unet_decoder.cuda()

    def forward(self, sketch):
        encoder_out = []
        next_sketch = sketch
        for enc_layer in self.basic_encoder.children():
            next_sketch = enc_layer(next_sketch)
            encoder_out.append(next_sketch)
        batch_size = sketch.shape[0]
        dec_output = torch.rand((batch_size, 0, 1, 1))
        if self.cuda:
            dec_output = dec_output.cuda()
        out_index = len(encoder_out) - 1
        for dec_layer in self.unet_decoder.children():
            enc_out = encoder_out[out_index]
            in_cat = torch.cat((enc_out, dec_output), 1)
            dec_output = dec_layer(in_cat)
            out_index -= 1
        return dec_output


class PatchDiscriminator70(nn.Module):
    def __init__(self, config):
        super(PatchDiscriminator70, self).__init__()
        self.network = nn.Sequential(ck_layer(config.in_channel, 64, norm=None),
                                     ck_layer(64, 128, config.norm),
                                     ck_layer(128, 256, config.norm),
                                     ck_layer(256, 512, config.norm, stride=1))
        self.network.add_module('out', nn.Sequential(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                                                     nn.Sigmoid()))
        self.cuda = config.cuda
        if config.cuda:
            self.network = self.network.cuda()

    def forward(self, input):
        conv = self.network(input)
        batchsize = conv.shape[0]
        out = torch.tensor([torch.mean(conv[i, :, :]) for i in range(batchsize)], requires_grad=True)
        if self.cuda:
            out = out.cuda()
        return out


class PixelDiscriminator(nn.Module):
    def __init__(self, config):
        super(PixelDiscriminator, self).__init__()
        self.conv1 = ck_layer(config.in_channel, 64, norm=None, kernel=1, stride=1, pad=0)
        self.conv2 = ck_layer(64, 128, config.norm, kernel=1, stride=1, pad=0)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, stride=1),
                                   nn.Sigmoid())
        if config.cuda:
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.conv3 = self.conv3.cuda()

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return torch.mean(out3)


class ImageDiscriminator(nn.Module):
    def __init__(self, config):
        super(ImageDiscriminator, self).__init__()
        self.network = nn.Sequential(ck_layer(config.in_channel, 64, norm=None),
                                     ck_layer(64, 128, config.norm),
                                     ck_layer(128, 256, config.norm),
                                     ck_layer(256, 512, config.norm),
                                     ck_layer(512, 512, config.norm),
                                     ck_layer(512, 512, config.norm, stride=1))
        self.network.add_module('out', nn.Sequential(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                                                     nn.Sigmoid()))
        if config.cuda:
            self.network=self.network.cuda()
    def forward(self, sketch, target):
        conv = self.network(input)
        return torch.mean(conv)


'''model'''


class Pix2Pix:
    def __init__(self, config, generator=generator_unet, discriminator=discriminator_patch):
        self.config = config
        if generator == generator_unet:
            self.generator = UnetGenerator(config)
        elif generator == generator_basic:
            self.generator = BasicGenerator(config)
        if discriminator == discriminator_patch:
            self.discriminator = PatchDiscriminator70(config)
        elif discriminator == discriminator_pixel:
            self.discriminator = PixelDiscriminator(config)
        elif discriminator == discriminator_image:
            self.discriminator = ImageDiscriminator(config)
        if config.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

    def train_model(self, dataloader):
        logpath = os.path.join(self.config.log_dir, '_' + self.config.model_name + '.log')
        logging.basicConfig(filename=logpath, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        self.generator.train()
        self.discriminator.train()
        d_optim = self.config.optimizer(self.discriminator.parameters())
        g_optim = self.config.optimizer(self.generator.parameters())
        ganloss = self.config.main_loss()
        for epoch in range(self.config.epoch):
            for step, (sketch, target) in enumerate(dataloader):
                sketch = sketch.float()
                target = target.float()
                if torch.cuda.is_available():
                    sketch = sketch.cuda()
                    target = target.cuda()
                d_optim.zero_grad()
                g_optim.zero_grad()
                generate = self.generator(sketch)
                if self.config.conditional:
                    discriminate_real = self.discriminator(torch.cat((sketch, target), 3))
                    discriminate_genera = self.discriminator(torch.cat((sketch, generate), 3))
                else:
                    discriminate_real = self.discriminator(target)
                    discriminate_genera = self.discriminator(generate)
                # optimize discriminator
                real_discrim_target = torch.ones(self.config.batch_size)
                gene_discrim_target = torch.zeros(self.config.batch_size)
                if self.config.cuda:
                    real_discrim_target = real_discrim_target.cuda()
                    gene_discrim_target = gene_discrim_target.cuda()
                d_loss = self.config.optim_coefficient_d * (
                        ganloss(discriminate_real, real_discrim_target)
                        + ganloss(discriminate_genera, gene_discrim_target))
                d_loss.backward()
                d_optim.step()
                # optimize generator
                discriminate_genera = self.discriminator(torch.cat((sketch, generate), 3))
                gene_target = torch.ones(self.config.batch_size)
                if self.config.cuda:
                    gene_target = gene_target.cuda()
                g_loss = ganloss(discriminate_genera, gene_target)
                g_loss += self.config.l1_coeficient * self.l1_loss(target, generate)
                g_loss.backward()
                g_optim.step()
                if step % 32 == 0:
                    logger.info('epoch' + str(epoch) + '-step' + str(step) + '-d_loss:' + str(d_loss.data))
                    logger.info('epoch' + str(epoch) + '-step' + str(step) + '-g_loss:' + str(g_loss.data))
                    if self.config.print_loss:
                        print('epoch' + str(epoch) + '-step' + str(step) + 'd_loss:' + str(d_loss.data))
                        print('epoch' + str(epoch) + '-step' + str(step) + 'g_loss:' + str(g_loss.data))
            if epoch % 32 == 0:
                self.store('epoch' + str(epoch))

    def l1_loss(self, target, generate):
        batch_size = self.config.batch_size
        abs = torch.abs(target - generate)
        loss = torch.sum(abs) / batch_size
        return loss

    def __call__(self, sketch):
        self.generator.eval()
        self.discriminator.eval()
        return self.generator(sketch)

    def store(self, info):
        info = '_' + info + '_'
        torch.save(self.generator.state_dict(),
                   os.path.join(self.config.save_dir, self.config.model_name + info + 'generator.pt'))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(self.config.save_dir, self.config.model_name + info + 'discriminator.pt'))
