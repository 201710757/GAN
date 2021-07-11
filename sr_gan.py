import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import itertools

from models.srgan import *
from models.datasets_srgan import *

n_epoch=1000
epoch = 0
dataset_name = "img_align_celeba"
batch_size = 4
lr = 1e-4 * 2
b1 = 0.5
b2 = 0.999
decay_epoch = 100
hr_height = 256
hr_width = 256
latent_dim = 100
img_size = 28
channels = 3
checkpoint_interval = 1
hr_shape = (hr_height, hr_width)
sample_interval = 100

generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(channels, *hr_shape))
feature_extractor = FeatureExtractor()

feature_extractor.eval()

#img_shape = (channels, img_size, img_size)
cuda = True if torch.cuda.is_available() else False

criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if epoch != 0:
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_model/discriminator_%d.pth"))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("./data/%s" % dataset_name, hr_shape=hr_shape),
    batch_size = batch_size,
    shuffle=True,
    num_workers=8,
)


import sys
for epoch in range(epoch, n_epoch):
    for i, (imgs, _) in enumerate(dataloader):
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False) 

        optimizer_G.zero_grad()
        gen_hr = generator(imgs_lr)

        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        loss_G = loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        sys.stdout.write(
            "Epoch[{}/{}] | Batch[{}/{}] | D loss : {} | G loss : {}".format(epoch, n_epoch, i, len(dataloader), loss_D.item(), loss_G.item())
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth"%epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
