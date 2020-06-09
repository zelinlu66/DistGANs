# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:26:08 2020

@authors: Andrey Prokpenko (e-mail: prokopenkoav@ornl.gov)
        : Debangshu Mukherjee (e-mail: mukherjeed@ornl.gov)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)
        : Vitaliy Starchenko  (e-mail: starchenkov@ornl.gov)
        : Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it) 

"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_features):
        super(Discriminator, self).__init__()
        self.hidden0 = nn.Sequential(nn.Linear(n_features, 250), nn.ReLU())
        self.hidden1 = nn.Sequential(nn.Linear(250, 100), nn.ReLU())
        # self.out = nn.Sequential(nn.Linear(100, 1), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(100, 1))

    def forward(self, input):
        x = self.hidden0(input)
        y = self.hidden1(x)
        z = self.out(y)
        return z

    def to(self,device):
        super(Discriminator,self).to(device)
        self.device = device

class Generator(torch.nn.Module):
    def __init__(self, noise_dimension, n_out):
        super(Generator, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(noise_dimension, 1000), nn.ReLU()
        )  #  ORIGINAL:   nn.LeakyReLU(0.2))
        self.hidden1 = nn.Sequential(nn.Linear(1000, 1000), nn.ReLU())
        self.hidden2 = nn.Sequential(nn.Linear(1000, 1000), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(1000, n_out), nn.Tanh())

    def forward(self, input):
        x = self.hidden0(input)
        y = self.hidden1(x)
        w = self.hidden2(y)
        z = self.out(w)
        return z

    def to(self,device):
        super(Generator,self).to(device)
        self.device = device

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor
        )


class GeneratorCNN(nn.Module):
    def __init__(self, noise_dimension, n_channels, image_dimension):
        super(GeneratorCNN, self).__init__()

        self.init_size = image_dimension // 4
        self.l1 = nn.Sequential(
            nn.Linear(noise_dimension, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            Upsample(scale_factor=2),  # nn.Upsample
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, n_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def to(self,device):
        super(GeneratorCNN,self).to(device)
        self.device = device

class DiscriminatorCNN(nn.Module):
    def __init__(self, n_channels, image_dimension):
        super(DiscriminatorCNN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(n_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = image_dimension // 2 ** 4
        self.adv_layer = nn.Sequential(
            # nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid()
            nn.Linear(128 * ds_size ** 2, 1)
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

    def to(self,device):
        super(DiscriminatorCNN,self).to(device)
        self.device = device
