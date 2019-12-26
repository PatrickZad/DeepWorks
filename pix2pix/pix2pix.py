import torch.nn as nn


class Pix2Pix():
    def __init__(self):
        pass

    def __basicGenerator(self):
        pass

    def train(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class BasicGenerator(nn.Module):
    def __init__(self):
        super.__init__(BasicGenerator, self)
        pass

    def forward(self, input):
        pass


class UnetGenerator(nn.Module):
    def __init__(self):
        super.__init__(UnetGenerator, self)
        pass

    def forward(self, input):
        pass


class BasicEncoder(nn.Module):
    def __init__(self):
        super.__init__(BasicEncoder, self)
        pass

    def forward(self, input):
        pass


class BasicDecoder(nn.Module):
    def __init__(self, outchannels=3):
        super.__init__(BasicDecoder, self)
        self.network = nn.Sequential(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=4),
                                     nn.BatchNorm2d(),
                                     nn.Dropout2d(0.5),
                                     nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=7),
                                     nn.Dropout2d(0.5),
                                     nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=13),
                                     nn.Dropout2d(0.5),
                                     nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=25),
                                     nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=49),
                                     nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=97),
                                     nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=193), )
        pass

    def forward(self, input):
        pass


class UnetDecoder(nn.Module):
    def __init__(self, outChannels=3):
        super.__init__(UnetDecoder, self)
        pass

    def forward(self, input):
        pass


discriminatorPatchSize = {'70by70': (70, 64),
                          '1by1': (1),
                          '16by16': (16),
                          '286by286': (286)}


class Discriminator(nn.Module):
    def __init__(self, patchsize):
        pass

    def forward(self, input):
        pass
