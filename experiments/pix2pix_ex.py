from model.pix2pix import pix2pix as pix2
import data.local_daterraset as datas
from torch.utils.data import DataLoader
import re, sys, os



if re.match(r'.*inux.*', sys.platform):
    storebase = r'/home/patrick/PatrickWorkspace/DeepWorks/store/pix2pix'
else:
    storebase = r''
cityscapes = datas.CityscapesTrain()
dataloader10 = DataLoader(cityscapes, batch_size=10, shuffle=True)
dataloader1 = DataLoader(cityscapes, batch_size=1, shuffle=True)
val = datas.CityscapesVal()
valloader = DataLoader(dataset=val, batch_size=len(val))
'''
loss
'''


def loss_ex():
    store_dir = os.path.join(storebase, 'loss')
    # gan
    logfile = r'./pix2pix_loss_gan.log'
    ganDiscriminator = pix2.PatchDiscriminator70()
    generator = pix2.UnetGenerator()
    model = pix2.Pix2Pix(generator, ganDiscriminator, cgan=False)
    model.trainGanModel(dataloader10, logfile, l1=0)
    model.store(store_dir, 'gan_only')
    # cgan
    logfile = r'./pix2pix_loss_cgan.log'
    cganDiscriminator = pix2.PatchDiscriminator70()
    generator = pix2.UnetGenerator()
    model = pix2.Pix2Pix(generator, cganDiscriminator)
    model.trainGanModel(dataloader10, logfile, l1=0)
    model.store(store_dir, 'cgan_only')
    # l1 gan
    logfile = r'./pix2pix_loss_l1gan.log'
    ganDiscriminator = pix2.PatchDiscriminator70(inchannels=0)
    generator = pix2.UnetGenerator()
    model = pix2.Pix2Pix(generator, ganDiscriminator, cgan=False)
    model.trainGanModel(dataloader10, logfile)
    model.store(store_dir, 'gan_l1')
    # l1 cgan
    logfile = r'./pix2pix_loss_l1cgan.log'
    cganDiscriminator = pix2.PatchDiscriminator70()
    generator = pix2.UnetGenerator()
    model = pix2.Pix2Pix(generator, cganDiscriminator)
    model.trainGanModel(dataloader10)
    model.store(store_dir, 'cgan_l1')


'''
generator
'''


def generator_ex():
    store_dir = os.path.join(storebase, 'generator')
    # ec
    ecgenerator = pix2.BasicGenerator()
    cganDiscriminator = pix2.PatchDiscriminator70()
    model = pix2.Pix2Pix(ecgenerator, cganDiscriminator)
    model.trainGanModel(dataloader10)
    model.store(store_dir, 'ec')
    # un
    ungenerator = pix2.UnetGenerator()
    cganDiscriminator = pix2.PatchDiscriminator70()
    model = pix2.Pix2Pix(ungenerator, cganDiscriminator)
    model.trainGanModel(dataloader10)
    model.store(store_dir, 'un')


'''
discriminator
'''


def discriminator_ex():
    store_dir = os.path.join(storebase, 'discriminator')
    # pixel
    generator = pix2.UnetGenerator()
    pixelDiscriminator = pix2.PixelDiscriminator()
    model = pix2.Pix2Pix(generator, pixelDiscriminator)
    model.trainGanModel(dataloader1)
    model.store(store_dir, 'pixel')
    # patch
    patchDiscriminator = pix2.PatchDiscriminator70()
    model = pix2.Pix2Pix(generator, patchDiscriminator)
    model.trainGanModel(dataloader1)
    model.store(store_dir, 'patch')
    # image
    imageDiscriminator = pix2.ImageDiscriminator()
    model = pix2.Pix2Pix(generator, imageDiscriminator)
    model.trainGanModel(dataloader1)
    model.store(store_dir, 'image')


if __name__ == '__main__':
    loss_ex()
    # generator_ex()
    # discriminator_ex()
