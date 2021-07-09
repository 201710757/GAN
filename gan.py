import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


n_epoch = 1000
batch_size = 32
lr = 1e-4
b1 = 0.5
b2 = 0.999
latent_dim = 100
img_size = 28
channels = 1
sample_interval = 400

img_shape = (channels, img_size, img_size)
cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.model = nn.Sequential(
                    *block(latent_dim, 128, normalize=False),
                    *block(128, 256),
                    *block(256, 512),
                    *block(512, 1024),
                    nn.Linear(1024, int(np.prod(img_shape))),
                    nn.Tanh()
                )

        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *img_shape)
            return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                    nn.Linear(int(np.prod(img_shape)), 512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
            return validity

adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversrial_loss.cuda()

import os
os.makedirs("./data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data/mnist",
                train=True,
                download=True,
                transform=transform.Compose(
                        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                    ),

                ), batch_size = batch_size, shuffle = True,
        )

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1,b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1,b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        optimizer_G.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()


        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), fake))
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print("Epoch {}/{} Batch {}/{} | D loss : {}  G loss : {}".format(epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(discriminator) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


    











