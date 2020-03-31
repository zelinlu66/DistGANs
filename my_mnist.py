# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:12:41 2020

@author: claud
"""

import torch
from modelli import *
from foo import *
from Dataloader import mnist_data
from optimizers import *
import matplotlib.pyplot as plt
from torch import nn, optim, autograd
from Log import Logger
import math
import timeit

data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)
lr = torch.tensor([0.0001])


generator = myGeneratorMNIST()
discriminator = myDiscriminatorMNIST()

criterion =  torch.nn.BCEWithLogitsLoss()


        
optimizer = myCGD_Jacobi(generator, discriminator, lr)

num_test_samples = 16
test_noise = noise(num_test_samples)

logger = Logger(model_name='VGAN', data_name='MNIST')


errorDreal = []
errorDfake = []
errorG = []

num_epochs = 100

start = time.time()

for epoch in range(num_epochs):
    e1 = 0.0
    e2 = 0.0
    e3 = 0.0
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        real_data = Variable(images_to_vectors(real_batch))
        fake_data = generator(noise(N))
        cg_x, cg_y, g_error, d_error, d_pred_real, d_pred_fake  = train(real_data, fake_data)

        index = 0
        for p in generator.parameters():
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in discriminator.parameters():
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0: 
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples, 
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
            
