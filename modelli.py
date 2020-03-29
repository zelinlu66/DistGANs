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
    
    
    

    
class myDiscriminator(nn.Module):
    def __init__(self):
        super(myDiscriminator, self).__init__()
        self.hidden0 = nn.Sequential(nn.Linear(784, 250),
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
    
class myGenerator(torch.nn.Module):
    
    def __init__(self):
        super(myGenerator, self).__init__()  
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
            nn.Linear(1000, 784),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.hidden0(input)
        y = self.hidden1(x)
        w = self.hidden2(y)
        z = self.out(w)
        return z
    
