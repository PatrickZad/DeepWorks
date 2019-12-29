import pix2pix.pix2pix as pix2
import data.local_dataset as datas
from torch.utils.data import DataLoader

cityscapes = datas.CityscapesTrain()
dataloader10 = DataLoader(cityscapes, batch_size=10, shuffle=True)
dataloader1 = DataLoader(cityscapes, batch_size=1, shuffle=True)
val = datas.CityscapesVal()
valloader = DataLoader(dataset=val, batch_size=len(val))
# loss

# gan
ganDiscriminator = pix2.PatchDiscriminator70(inchannels=0)
generator = pix2.UnetGenerator()
model = pix2.Pix2Pix(generator, ganDiscriminator, cgan=False)
model.trainGanModel(dataloader10, l1=0)
# cgan
cganDiscriminator = pix2.PatchDiscriminator70()
generator = pix2.UnetGenerator()
model = pix2.Pix2Pix(generator, ganDiscriminator)
model.trainGanModel(dataloader10, l1=0)
# l1 gan
ganDiscriminator = pix2.PatchDiscriminator70(inchannels=0)
generator = pix2.UnetGenerator()
model = pix2.Pix2Pix(generator, ganDiscriminator, cgan=False)
model.trainGanModel(dataloader10)
# l1 cgan
cganDiscriminator = pix2.PatchDiscriminator70()
generator = pix2.UnetGenerator()
model = pix2.Pix2Pix(generator, ganDiscriminator)
model.trainGanModel(dataloader10)
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
