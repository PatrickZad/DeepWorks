import torch
import torch.nn as nn
from collections import OrderedDict


class Pix2Pix():
    def __init__(self):
        pass

    def __basicGenerator(self):
        pass

    def trainModel(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class BasicGenerator(nn.Module):
    def __init__(self, inchannels=3, outchannels=3):
        super.__init__(BasicGenerator, self)
        self.basic_encoder = BasicEncoder(inchannels)
        self.basic_decoder = BasicDecoder(outchannels)

    def forward(self, input):
        encode = self.basic_encoder.encoder(input)
        decode = self.basic_decoder.decoder(encode)
        return self.basic_decoder.out(decode)


class UnetGenerator(nn.Module):
    def __init__(self, inchannels=3, outchannels=3):
        super.__init__(UnetGenerator, self)
        self.basic_encoder = BasicEncoder(inchannels)
        self.unet_decoder = UnetDecoder(outchannels)

    def forward(self, input):
        encoderout = []
        nextinput = input
        for enclayer in self.basic_encoder.children():
            nextinput = enclayer(nextinput)
            encoderout.append(nextinput)
        decoutput = torch.rand((0, 1, 1))
        for declayer, encout in self.unet_decoder.children(), encoderout.reverse():
            incat = torch.cat((encout, nextinput), 0)
            decoutput = declayer(incat)
        return decoutput


def cklayer(inChannel, outChannel, ):
    conv = nn.Conv2d(inChannel, outChannel, kernel_size=4, padding=1, stride=2)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(
        conv,
        bn,
        nn.LeakyReLU(0.2))


def transpose_cklayer(inChannel, outChannel):
    conv = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=4, padding=1, stride=2)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(
        conv,
        bn,
        nn.ReLU())


def cdklayer(inChannel, outChannel):
    conv = nn.Conv2d(inChannel, outChannel, kernel_size=4, padding=1, stride=2)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(
        conv,
        bn,
        nn.Dropout2d(0.5),
        nn.LeakyReLU(0.2))


def transpose_cdklayer(inChannel, outChannel):
    conv = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=4, padding=1, stride=2)
    nn.init.normal(conv.weight.data, mean=0, std=0.02)
    bn = nn.BatchNorm2d(outChannel)
    nn.init.normal(bn.weight.data, mean=0, std=0.02)
    nn.init.constant(bn.bias.data, 0)
    return nn.Sequential(
        conv,
        bn,
        nn.Dropout2d(0.5),
        nn.ReLU())


class BasicEncoder():
    def __init__(self, inchannels=3):
        super.__init__(BasicEncoder, self)
        self.encoder = nn.Sequential(
            OrderedDict([('enc1', nn.Sequential(nn.Conv2d(inchannels, 64, kernel_size=4, stride=2, padding=1),
                                                nn.LeakyReLU(0.2))),
                         ('enc2', cklayer(64, 128)),
                         ('enc3', cklayer(128, 256)),
                         ('enc4', cklayer(256, 512)),
                         ('enc5', cklayer(512, 512)),
                         ('enc6', cklayer(512, 512)),
                         ('enc7', cklayer(512, 512))]))


class BasicDecoder():
    def __init__(self, outchannels=3):
        super.__init__(BasicDecoder, self)
        self.decoder = nn.Sequential(transpose_cdklayer(512, 512),
                                     transpose_cdklayer(512, 512),
                                     transpose_cdklayer(512, 512),
                                     transpose_cklayer(512, 512),
                                     transpose_cklayer(512, 256),
                                     transpose_cklayer(256, 128),
                                     transpose_cklayer(128, 64))
        self.out = nn.Sequential(nn.ConvTranspose2d(64, outchannels, kernel_size=4, stride=2, padding=1),
                                 nn.Tanh())


class UnetDecoder():
    def __init__(self, outChannels=3):
        super.__init__(UnetDecoder, self)
        self.decoder = nn.Sequential(
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


discriminatorPatchSize = {'70by70': (64, 128, 256, 512),
                          '1by1': (64, 128),
                          '16by16': (64, 128),
                          '286by286': (64, 128, 256, 512, 512, 512)}


class Discriminator(nn.Module):
    def __init__(self, patchsize='70by70', inchannels=3):
        architecture = discriminatorPatchSize[patchsize]
        if patchsize is not '1by1':
            self.add_module('dis0', nn.Sequential(nn.Conv2d(inchannels, kernel_size=4, stride=2, padding=1),
                                                  nn.LeakyReLU(0.2)))
            for i in range(1,len(architecture)):
                self.add_module()
        pass

    def forward(self, input):
        pass
