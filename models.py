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
    
    
class Discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, n_features):
        super(Discriminator, self).__init__()
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
 
    
    
class Generator(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, noise_dimension, n_out):
        super(Generator, self).__init__()
        
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

class myDiscriminatorCIFAR10(nn.Module):
    def __init__(self):
        super(myDiscriminatorCIFAR10, self).__init__()
        self.hidden0 = nn.Sequential(nn.Linear(3072, 250),
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
    
class myGeneratorCIFAR10(torch.nn.Module):
    
    def __init__(self):
        super(myGeneratorCIFAR10, self).__init__()  
        self.hidden0 = nn.Sequential(
            nn.Linear(100, 1000),
            nn.LeakyReLU(0.2))
        self.hidden1 = nn.Sequential(
                nn.Linear(1000,1000),
                nn.ReLU())
        self.hidden2 = nn.Sequential(
                nn.Linear(1000,1000),
                nn.ReLU())
        self.out = nn.Sequential(
            nn.Linear(1000, 3072),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.hidden0(input)
        y = self.hidden1(x)
        w = self.hidden2(y)
        z = self.out(w)
        return z
