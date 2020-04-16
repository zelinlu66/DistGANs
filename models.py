# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:26:08 2020

@author: claud
"""
import torch
import torch.nn as nn

# GoodGenerator and GoodDiscriminator by Florian Shafer for CIFAR10
# MyGenerator and MyDiscriminator by me for MNIST

DIM = 64


class GoodGenerator(nn.Module):
    def __init__(self):
        super(GoodGenerator, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            # nn.Softplus(),
            nn.ConvTranspose2d(2 * DIM, DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
            # nn.Softplus(),
            nn.ConvTranspose2d(DIM, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.main_module(output)
        return output.view(-1, 3, 32, 32)


class GoodDiscriminator(nn.Module):
    def __init__(self):
        super(GoodDiscriminator, self).__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(3, DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 16x16
            nn.Conv2d(DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 8x8
            nn.Conv2d(2 * DIM, 4 * DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 4 x 4
        )
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main_module(input)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        # print(output.shape)
        output = self.linear(output)
        # print(output.shape)
        return output
    
    
class Discriminator2(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, n_features):
        super(Discriminator2, self).__init__()
        n_out = 1
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
 
    
    
class Generator2(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, noise_dimension, n_out):
        super(Generator2, self).__init__()
        
        self.hidden0 = nn.Sequential(
            nn.Linear(noise_dimension, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
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
            nn.LeakyReLU(0.2))
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
    
    
class DiscriminativeNet(torch.nn.Module):
    def __init__(self, n_channels):
        super(DiscriminativeNet, self).__init__()
        
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


class GenerativeNet(torch.nn.Module):
    
    def __init__(self, noise_dimension, n_channels):
        super(GenerativeNet, self).__init__()
        
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
    
