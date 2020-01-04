import torch
import torch.nn as nn
from collections import OrderedDict
import os
import logging
from model import NNconfig

'''config'''
#constant
batch_norm = 0
inst_norm = 1


class Config(NNconfig):
    def __init__(self, norm=None):
        self.lr = 2 * 10 ** (-4)
        self.momentum_beta1 = 0.5
        self.momentum_beta2 = 0.999
        self.norm = norm
        self.optim_coefficient_d = 0.5
        self.optim_coefficient_g = 1
        if norm == inst_norm:
            self.batch_size = 1
        else:
            self.batch_size = 10
        self.cuda = torch.cuda.is_available()

    def optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.lr, betas=(self.momentum_beta1, self.momentum_beta2))

    def main_loss(self):
        return nn.BCELoss()


'''build functions'''


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


def basic_encoder(norm, in_channels=3, ):
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


'''other functions'''


def L1_loss(target, generate):
    batchsize = target.shape[0]
    abs = torch.abs(target - generate)
    loss = torch.sum(abs) / batchsize
    return loss


class Pix2Pix:
    def __init__(self, config):
        pass

    def __init__(self, generator, discriminator, cgan=True):
        self.generator = generator
        self.discriminator = discriminator
        self.cgan = cgan

    def trainGanModel(self, dataloader, logfile, l1=100):
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        self.generator.train()
        self.discriminator.train()
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        g_optim = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        ganloss = nn.BCELoss()
        for epoch in range(2000):
            for step, (sketch, target) in enumerate(dataloader):
                # sketch = torch.from_numpy(sketch)
                # target = torch.from_numpy(target)
                sketch = sketch.float()
                target = target.float()
                '''
                if torch.cuda.is_available():
                    sketch = sketch.cuda()
                    target = target.cuda()'''
                d_optim.zero_grad()
                g_optim.zero_grad()
                batchsize = sketch.shape[0]
                generate = self.generator(sketch)
                if self.cgan:
                    discriminate_real = self.discriminator(torch.cat((sketch, target), 3))
                    discriminate_genera = self.discriminator(torch.cat((sketch, generate), 3))
                else:
                    discriminate_real = self.discriminator(target)
                    discriminate_genera = self.discriminator(generate)
                # optimize discriminator
                d_loss = 0.5 * (ganloss(discriminate_real, torch.ones(batchsize))
                                + ganloss(discriminate_genera, torch.zeros(batchsize)))
                logger.info(str(epoch)+'-'+str(step)+'-d_loss:' + str(d_loss.data))
                d_loss.backward()
                d_optim.step()
                # optimize generator
                discriminate_genera = self.discriminator(torch.cat((sketch, generate), 3))
                g_loss = ganloss(discriminate_genera, torch.ones(batchsize))
                g_loss += l1 * L1_loss(target, generate)
                logger.info(str(epoch)+'-'+str(step)+'-g_loss:' + str(g_loss.data))
                g_loss.backward()
                g_optim.step()

    def __call__(self, sketch):
        self.generator.eval()
        self.discriminator.eval()
        return self.generator(sketch)

    def store(self, dir, name):
        torch.save(self.generator.stat_dict(), os.path.join(dir, name + '_generator.pt'))
        torch.save(self.discriminator.stat_dict(), os.path.join(dir, name + '_discriminator.pt'))


class BasicGenerator(nn.Module):
    def __init__(self, inchannels=3, outchannels=3):
        super(BasicGenerator, self).__init__()
        self.basic_encoder = basic_encoder(inchannels)
        self.basic_decoder = basic_decoder(outchannels)

    def forward(self, sketch):
        encode = self.basic_encoder.encoder(sketch)
        decode = self.basic_decoder.decoder(encode)
        return self.basic_decoder.out(decode)


class UnetGenerator(nn.Module):
    def __init__(self, inchannels=3, outchannels=3):
        super(UnetGenerator, self).__init__()
        self.basic_encoder = basic_encoder(inchannels)
        self.unet_decoder = unet_decoder(outchannels)

    def forward(self, sketch):
        encoderout = []
        nextsketch = sketch
        for enc_layer in self.basic_encoder.children():
            nextsketch = enc_layer(nextsketch)
            encoderout.append(nextsketch)
        batchsize = sketch.shape[0]
        dec_output = torch.rand((batchsize, 0, 1, 1))
        outindex = len(encoderout) - 1
        for declayer in self.unet_decoder.children():
            encout = encoderout[outindex]
            incat = torch.cat((encout, dec_output), 1)
            dec_output = declayer(incat)
            outindex -= 1
        return dec_output


class PatchDiscriminator70(nn.Module):
    def __init__(self, inchannels=3):
        super(PatchDiscriminator70, self).__init__()
        self.network = nn.Sequential(cklayer_no_bn(inchannels, 64),
                                     cklayer(64, 128),
                                     cklayer(128, 256),
                                     cklayer(256, 512, stride=1))
        self.network.add_module('out', nn.Sequential(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                                                     nn.Sigmoid()))

    def forward(self, input):
        conv = self.network(input)
        batchsize = conv.shape[0]
        out = torch.tensor([torch.mean(conv[i, :, :]) for i in range(batchsize)], requires_grad=True)
        return out.squeeze()


class PixelDiscriminator(nn.Module):
    def __init__(self, inchannels=3):
        super(PixelDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inchannels, 64, kernel_size=1, stride=1),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, stride=1),
                                   nn.Sigmoid())

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return torch.mean(out3)


class ImageDiscriminator(nn.Module):
    def __init__(self, inchannels=3):
        super(ImageDiscriminator, self).__init__()
        self.network = nn.Sequential(cklayer_no_bn(inchannels, 64),
                                     cklayer(64, 128),
                                     cklayer(128, 256),
                                     cklayer(256, 512),
                                     cklayer(512, 512),
                                     cklayer(512, 512, stride=1))
        self.network.add_module('out', nn.Sequential(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                                                     nn.Sigmoid()))

    def forward(self, sketch, target):
        conv = self.network(input)
        return torch.mean(conv)
