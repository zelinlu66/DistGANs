# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:26:08 2020

@authors: Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)

"""
import torch
import torch.nn as nn
    
class Discriminator(nn.Module):
    def __init__(self, n_features):
        super(Discriminator, self).__init__()
        self.hidden0 = nn.Sequential(nn.Linear(n_features, 250),
                                     nn.ReLU())
        self.hidden1 = nn.Sequential(nn.Linear(250,100),
                                     nn.ReLU())
        self.out = nn.Sequential(nn.Linear(100,1),
                                 nn.Sigmoid())
    def forward(self, input):
        x = self.hidden0(input)
        y = self.hidden1(x)
        z = self.out(y)
        return z
    
class Generator(torch.nn.Module):
    
    def __init__(self,noise_dimension, n_out):
        super(Generator, self).__init__()  
        self.hidden0 = nn.Sequential(
            nn.Linear(noise_dimension, 1000),
            nn.ReLU())                      #  ORIGINAL:   nn.LeakyReLU(0.2))
        self.hidden1 = nn.Sequential(
                nn.Linear(1000,1000),
                nn.ReLU())
        self.hidden2 = nn.Sequential(
                nn.Linear(1000,1000),
                nn.ReLU())
        self.out = nn.Sequential(
            nn.Linear(1000, n_out),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.hidden0(input)
        y = self.hidden1(x)
        w = self.hidden2(y)
        z = self.out(w)
        return z
    
    
class DiscriminativeCNN(torch.nn.Module):
    def __init__(self, n_channels):
        super(DiscriminativeCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels, out_channels=128, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024*4*4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024*4*4)
        x = self.out(x)
        return x


class GenerativeCNN(torch.nn.Module):
    
    def __init__(self, noise_dimension, n_channels):
        super(GenerativeCNN, self).__init__()
        
        self.linear = torch.nn.Linear(noise_dimension, 1024*4*4)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=n_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)
    
