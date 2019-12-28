import pix2pix.pix2pix as pix2
import data.local_dataset as datas
from torch.utils.data import DataLoader

cityscapes = datas.CityscapesTrain()
dataloader10 = DataLoader(cityscapes, batch_size=10, shuffle=True)
dataloader1 = DataLoader(cityscapes, batch_size=1, shuffle=True)

# loss
generator = pix2.UnetGenerator()
# gan
ganDiscriminator = pix2.PatchDiscriminator70(inchannels=0)
# cgan
cganDiscriminator = pix2.PatchDiscriminator70()
# l1 gan
# l1 cgan
# generator
# ec
ecgenerator = pix2.BasicGenerator()
# un
ungenerator = pix2.UnetGenerator()
# discriminator
generator = pix2.UnetGenerator()
pixelDiscriminator = pix2.PixelDiscriminator()
patchDiscriminator = pix2.PatchDiscriminator70()
imageDiscriminator = pix2.ImageDiscriminator()
