import torch
import torch.nn as nn
from collections import OrderedDict


class Pix2Pix:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def trainModel(self, dataloader, l1_loss=True):
        self.generator.train()
        self.discriminator.train()
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        g_optim = torch.optim.Adam(self.generator.parameters, lr=0.0002, betas=(0.5, 0.999))
        for epoch in range(500):
            for step, (sketch, target) in enumerate(dataloader):
                if torch.cuda.is_available():
                    sketch = sketch.cuda()
                    target = target.cuda()
                d_optim.zero_grad()
                g_optim.zero_grad()
                batchsize = sketch.shape[0]
                generate = self.generator(sketch)
                discriminate_real = self.discriminator(sketch, target)
                discriminate_genera = self.discriminator(sketch, generate)
                # optimize discriminator
                d_loss = 0.5 * (cGAN_loss(discriminate_real, torch.ones(batchsize))
                                + cGAN_loss(discriminate_genera, torch.zeros(batchsize)))
                d_loss.backward()
                d_optim.step()
                # optimize generator
                discriminate_genera = self.discriminator(sketch, generate)
                g_loss = cGAN_loss(discriminate_genera, torch.ones(batchsize))
                if l1_loss:
                    g_loss += L1_loss(target, generate)
                g_loss.backward()
                g_optim.step()

    def __call__(self, sketch):
        self.generator.eval()
        self.discriminator.eval()
        return self.generator(sketch)


def cGAN_loss(sketch, target):
    return nn.BCELoss(sketch, target)


def L1_loss(target, generate):
    batchsize = target.shape[0]
    abs = torch.abs(target - generate)
    return torch.sum(abs) / batchsize


class BasicGenerator(nn.Module):
    def __init__(self, inchannels=3, outchannels=3):
        super.__init__(BasicGenerator, self)
        self.basic_encoder = basic_encoder(inchannels)
        self.basic_decoder = basic_decoder(outchannels)

    def forward(self, sketch):
        encode = self.basic_encoder.encoder(sketch)
        decode = self.basic_decoder.decoder(encode)
        return self.basic_decoder.out(decode)


class UnetGenerator(nn.Module):
    def __init__(self, inchannels=3, outchannels=3):
        super.__init__(UnetGenerator, self)
        self.basic_encoder = basic_encoder(inchannels)
        self.unet_decoder = unet_decoder(outchannels)

    def forward(self, sketch):
        encoderout = []
        nextsketch = sketch
        for enc_layer in self.basic_encoder.children():
            nextsketch = enc_layer(nextsketch)
            encoderout.append(nextsketch)
        dec_output = torch.rand((0, 1, 1))
        for declayer, encout in self.unet_decoder.children(), encoderout.reverse():
            incat = torch.cat((encout, dec_output), 0)
            dec_output = declayer(incat)
        return dec_output


def cklayer(inChannel, outChannel, stride=2):
    conv = nn.Conv2d(inChannel, outChannel, kernel_size=4, padding=1, stride=stride)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(conv,
                         bn,
                         nn.LeakyReLU(0.2))


def cklayer_no_bn(inChannel, outChannel, stride=2):
    conv = nn.Conv2d(inChannel, outChannel, kernel_size=4, padding=1, stride=stride)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    return nn.Sequential(conv,
                         nn.LeakyReLU(0.2))


def transpose_cklayer(inChannel, outChannel, stride=2):
    conv = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=4, padding=1, stride=stride)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(
        conv,
        bn,
        nn.ReLU())


def cdklayer(inChannel, outChannel, stride=2):
    conv = nn.Conv2d(inChannel, outChannel, kernel_size=4, padding=1, stride=stride)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(
        conv,
        bn,
        nn.Dropout2d(0.5),
        nn.LeakyReLU(0.2))


def transpose_cdklayer(inChannel, outChannel, stride=2):
    conv = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=4, padding=1, stride=stride)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(
        conv,
        bn,
        nn.Dropout2d(0.5),
        nn.ReLU())


def basic_encoder(inchannels=3):
    return nn.Sequential(
        OrderedDict([('enc1', cklayer_no_bn(inchannels, 64)),
                     ('enc2', cklayer(64, 128)),
                     ('enc3', cklayer(128, 256)),
                     ('enc4', cklayer(256, 512)),
                     ('enc5', cklayer(512, 512)),
                     ('enc6', cklayer(512, 512)),
                     ('enc7', cklayer(512, 512))]))


def basic_decoder(outchannels=3):
    return nn.Sequential(transpose_cdklayer(512, 512),
                         transpose_cdklayer(512, 512),
                         transpose_cdklayer(512, 512),
                         transpose_cklayer(512, 512),
                         transpose_cklayer(512, 256),
                         transpose_cklayer(256, 128),
                         transpose_cklayer(128, 64),
                         nn.Sequential(nn.ConvTranspose2d(64, outchannels, kernel_size=4, stride=2, padding=1),
                                       nn.Tanh()))


def unet_decoder(outChannels=3):
    return nn.Sequential(
        OrderedDict([('dec-7', transpose_cdklayer(512, 512)),
                     ('dec-6', transpose_cdklayer(1024, 512)),
                     ('dec-5', transpose_cdklayer(1024, 512)),
                     ('dec-4', transpose_cdklayer(1024, 512)),
                     ('dec-3', transpose_cdklayer(1024, 256)),
                     ('dec-2', transpose_cdklayer(512, 128)),
                     ('dec-1', transpose_cdklayer(256, 64)),
                     ('decout',
                      nn.Sequential(nn.ConvTranspose2d(128, outChannels, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh()))]))


class PatchDiscriminator70(nn.Module):
    def __init__(self, inchannels=3, targetchannels=3):
        super.__init__(PatchDiscriminator70, self)
        self.network = nn.Sequential(cklayer_no_bn(inchannels + targetchannels, 64),
                                     cklayer(64, 128),
                                     cklayer(128, 256),
                                     cklayer(256, 512, stride=1))
        self.network.add_module('out', nn.Sequential(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                                                     nn.Sigmoid()))

    def forward(self, sketch, target):
        cat = nn.cat((sketch, target), 0)
        conv = self.network(cat)
        return torch.mean(conv)


class PixelDiscriminator(nn.Module):
    def __init__(self, inchannels=3, targetchannels=3):
        super.__init__(PixelDiscriminator, self)
        self.conv1 = nn.Sequential(nn.Conv2d(inchannels + targetchannels, 64, kernel_size=1, stride=1),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, stride=1),
                                   nn.Sigmoid())

    def forward(self, sketch, target):
        cat = torch.cat((sketch, target), 0)
        out1 = self.conv1(cat)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return torch.mean(out3)


class ImageDiscriminator(nn.Module):
    def __init__(self, inchannels=3, targetchannels=3):
        super.__init__(ImageDiscriminator, self)
        # TODO

    def forward(self, sketch, target):
        # TODO
        pass
