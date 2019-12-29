import torch
import torch.nn as nn
from collections import OrderedDict
import os


class Pix2Pix:
    def __init__(self, generator, discriminator, cgan=True):
        self.generator = generator
        self.discriminator = discriminator
        self.cgan = cgan

    def trainGanModel(self, dataloader, l1=100):
        self.generator.train()
        self.discriminator.train()
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        g_optim = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        ganloss = nn.BCELoss()
        for epoch in range(500):
            for step, (sketch, target) in enumerate(dataloader):
                if torch.cuda.is_available():
                    sketch = sketch.cuda()
                    target = target.cuda()
                d_optim.zero_grad()
                g_optim.zero_grad()
                batchsize = sketch.shape[0]
                generate = self.generator(sketch)
                if self.cgan:
                    discriminate_real = self.discriminator(torch.cat((sketch, target), 0))
                    discriminate_genera = self.discriminator(torch.cat((sketch, generate), 0))
                else:
                    discriminate_real = self.discriminator(target)
                    discriminate_genera = self.discriminator(generate)
                # optimize discriminator
                d_loss = 0.5 * (ganloss(discriminate_real, torch.ones(batchsize))
                                + ganloss(discriminate_genera, torch.zeros(batchsize)))
                d_loss.backward()
                d_optim.step()
                # optimize generator
                discriminate_genera = self.discriminator(sketch, generate)
                g_loss = ganloss(discriminate_genera, torch.ones(batchsize))
                g_loss += l1 * L1_loss(target, generate)
                g_loss.backward()
                g_optim.step()

    def __call__(self, sketch):
        self.generator.eval()
        self.discriminator.eval()
        return self.generator(sketch)

    def store(self, dir,name):
        torch.save(self.generator.stat_dict(), os.path.join(dir, name+'_generator.pt'))
        torch.save(self.discriminator.stat_dict(), os.path.join(dir, name+'_discriminator.pt'))


def L1_loss(target, generate):
    batchsize = target.shape[0]
    abs = torch.abs(target - generate)
    return torch.sum(abs) / batchsize


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
        dec_output = torch.rand((0, 1, 1))
        for declayer, encout in self.unet_decoder.children(), encoderout.reverse():
            incat = torch.cat((encout, dec_output), 0)
            dec_output = declayer(incat)
        return dec_output


def cklayer(inChannel, outChannel, kernel=4, stride=2, pad=1):
    conv = nn.Conv2d(inChannel, outChannel, kernel_size=kernel, padding=pad, stride=stride)
    nn.init.normal_(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal_(bn.weight.data, mean=0, std=0.02)
    nn.init.constant_(bn.bias.data, 0)
    return nn.Sequential(conv,
                         bn,
                         nn.LeakyReLU(0.2))


def cklayer_no_bn(inChannel, outChannel, kernel=4, stride=2, pad=1):
    conv = nn.Conv2d(inChannel, outChannel, kernel_size=kernel, padding=pad, stride=stride)
    nn.init.normal_(conv.weight.data, mean=0, std=0.02)
    return nn.Sequential(conv,
                         nn.LeakyReLU(0.2))


def transpose_cklayer(inChannel, outChannel, kernel=4, stride=2, pad=1):
    conv = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=kernel, padding=pad, stride=stride)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(
        conv,
        bn,
        nn.ReLU())


def cdklayer(inChannel, outChannel, kernel=4, stride=2, pad=1):
    conv = nn.Conv2d(inChannel, outChannel, kernel_size=kernel, padding=pad, stride=stride)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(
        conv,
        bn,
        nn.Dropout2d(0.5),
        nn.LeakyReLU(0.2))


def transpose_cdklayer(inChannel, outChannel, kernel=4, stride=2, pad=1):
    conv = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=kernel, padding=pad, stride=stride)
    nn.init.normal_(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal_(bn.weight.data, mean=0, std=0.02)
    nn.init.constant_(bn.bias.data, 0)
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
        super(PatchDiscriminator70, self).__init__()
        self.network = nn.Sequential(cklayer_no_bn(inchannels + targetchannels, 64),
                                     cklayer(64, 128),
                                     cklayer(128, 256),
                                     cklayer(256, 512, stride=1))
        self.network.add_module('out', nn.Sequential(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                                                     nn.Sigmoid()))

    def forward(self, input):
        conv = self.network(input)
        return torch.mean(conv)


class PixelDiscriminator(nn.Module):
    def __init__(self, inchannels=3, targetchannels=3):
        super(PixelDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inchannels + targetchannels, 64, kernel_size=1, stride=1),
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
    def __init__(self, inchannels=3, targetchannels=3):
        super(ImageDiscriminator, self).__init__()
        self.network = nn.Sequential(cklayer_no_bn(inchannels + targetchannels, 64),
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
