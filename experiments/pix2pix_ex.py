from model.pix2pix.pix2pix import ModelConfig, Pix2Pix, batch_norm, generator_basic, discriminator_image, \
    discriminator_pixel
from data.local_dataset import FacadesTrain, FacadesVal, CityscapesTrain, CityscapesVal, MapTrain, MapVal
from torch.utils.data import DataLoader
import re, sys, os
from experiments import platform_kaggle, platform_linux, platform_win, platform_kaggle_test, ExperimentConfig

'''train'''
# ex_config = ExperimentConfig(platform_linux)
ex_config = ExperimentConfig(platform_kaggle_test)


def loss_ex():
    # loss experiments with facades
    facades_train = FacadesTrain(ex_config.platform)
    # cgan with l1
    cgan_l1_config = ModelConfig(ex_config)
    cgan_l1_config.model_name = 'loss_cgan_l1'
    # dataloder = DataLoader(facades_train, batch_size=cgan_l1_config.batch_size, shuffle=True)
    cgan_l1_model = Pix2Pix(cgan_l1_config)
    cgan_l1_model.train_model(facades_train)
    # gan only
    gan_only_config = ModelConfig(ex_config)
    gan_only_config.conditional = False
    gan_only_config.model_name = 'loss_gan_only'
    gan_only_config.l1_coeficient = 0
    # dataloder = DataLoader(facades_train, batch_size=gan_only_config.batch_size, shuffle=True)
    gan_only_model = Pix2Pix(gan_only_config)
    gan_only_model.train_model(facades_train)
    # cgan only
    cgan_only_config = ModelConfig(ex_config)
    cgan_only_config.model_name = 'loss_cgan_only'
    cgan_only_config.l1_coeficient = 0
    # dataloder = DataLoader(facades_train, batch_size=cgan_only_config.batch_size, shuffle=True)
    cgan_only_model = Pix2Pix(cgan_only_config)
    cgan_only_model.train_model(facades_train)
    # gan with l1
    gan_l1_config = ModelConfig(ex_config)
    gan_l1_config.conditional = False
    gan_l1_config.model_name = 'loss_gan_l1'
    # dataloder = DataLoader(facades_train, batch_size=gan_l1_config.batch_size, shuffle=True)
    gan_l1_model = Pix2Pix(gan_l1_config)
    gan_l1_model.train_model(facades_train)


def generator_ex():
    # generator experiments with cityscapes
    cityscapes_train = CityscapesTrain(ex_config.platform)
    # basic generator
    basic_generator_config = ModelConfig(ex_config)
    basic_generator_config.norm = batch_norm
    basic_generator_config.batch_size = 10
    basic_generator_config.model_name = 'generator_basic'
    # dataloader = DataLoader(cityscapes_train, batch_size=basic_generator_config.batch_size, shuffle=True)
    basic_generator_model = Pix2Pix(basic_generator_config, generator_basic)
    basic_generator_model.train_model(cityscapes_train)
    # unet generator
    unet_generator_config = ModelConfig(ex_config)
    unet_generator_config.norm = batch_norm
    unet_generator_config.batch_size = 10
    unet_generator_config.model_name = 'generator_unet'
    # dataloader = DataLoader(cityscapes_train, batch_size=basic_generator_config.batch_size, shuffle=True)
    unet_generator_model = Pix2Pix(basic_generator_config, generator_basic)
    unet_generator_model.train_model(cityscapes_train)


def discriminator_ex():
    # discriminator experiments with maps
    maps_train = MapTrain(ex_config.platform)
    # patch discriminator
    patch_config = ModelConfig(ex_config)
    patch_config.model_name = 'patch_model'
    dataloader = DataLoader(maps_train, batch_size=patch_config.batch_size, shuffle=True)
    patch_model = Pix2Pix(patch_config)
    patch_model.train_model(dataloader)
    # pixel discriminator
    pixel_config = ModelConfig(ex_config)
    pixel_config.model_name = 'pixel_model'
    dataloader = DataLoader(maps_train, batch_size=pixel_config.batch_size, shuffle=True)
    pixel_model = Pix2Pix(pixel_config, discriminator=discriminator_pixel)
    pixel_model.train_model(dataloader)
    # image discriminator
    image_config = ModelConfig(ex_config)
    image_config.model_name = 'image_model'
    dataloader = DataLoader(maps_train, batch_size=image_config.batch_size, shuffle=True)
    image_model = Pix2Pix(image_config, discriminator=discriminator_image)
    image_model.train_model(dataloader)


'''val'''
model_dir = ['/home/patrick/PatrickWorkspace/DeepWorks/out', '', '']
out_dir = ['/home/patrick/PatrickWorkspace/DeepWorks/out', '', '']


def val_generator(binary, data_val):
    from model.pix2pix.pix2pix import UnetGenerator
    import torch
    import cv2
    import skimage.io as imgio
    config = ModelConfig(ex_config)
    generator = UnetGenerator(config)
    generator.load_state_dict(torch.load(os.path.join(model_dir[0], binary), map_location='cpu'))
    dataloader = DataLoader(data_val, batch_size=1)
    for index, (sketch, target) in enumerate(dataloader):
        sketch = sketch.float()
        target = target.float()
        generate = generator(sketch)
        # result = torch.cat((generate, target), dim=0).squeeze()
        result = generate.squeeze()
        array = result.detach().numpy().transpose(1, 2, 0)
        imgio.imsave(os.path.join(out_dir[0], str(index) + '.jpg'), array)


def quantitative(generate, target):
    pass


'''loss experiments'''

'''generator experiments'''

'''discriminator experiments'''
'''
loss
'''

'''
generator
'''

if __name__ == '__main__':
    loss_ex()
    # generator_ex()
    # discriminator_ex()
    # val
    #binary = 'loss_cgan_l1_epoch224_generator.pt'
    #data = CityscapesVal(platform_linux)
    #val_generator(binary, data)
