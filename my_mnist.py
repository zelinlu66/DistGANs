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


num_epochs = 5

for e in range(num_epochs):
            for n_batch, (real_batch,_) in enumerate(data_loader):
                N = real_batch.size(0)
                real_data = Variable(images_to_vectors(real_batch))
                fake_data = generator(noise(N))
                
                d_pred_real = discriminator(real_data)
                error_real = criterion(d_pred_real, ones_target(N) )
                d_pred_fake = discriminator(fake_data)
                error_fake = criterion(d_pred_fake, zeros_target(N))
                g_error = criterion(d_pred_fake, ones_target(N))
                d_error = error_real
                #f = d_error
                #g  = error_fake
                loss = error_fake + error_real
                #loss = d_pred_real.mean() - d_pred_fake.mean()

                optimizer.zero_grad()
                optimizer.step(loss)
                iter_num = -1
                # Log batch error
                logger.log(d_error, g_error, e, n_batch, num_batches)
                # Display Progress every few batches
                if (n_batch) % 100 == 0: 
                    test_images = vectors_to_images(generator(test_noise))
                    test_images = test_images.data
                    logger.log_images(
                            test_images, num_test_samples, 
                            e, n_batch, num_batches
                            );
                            # Display status Logs
                    logger.display_status(
                                    e, num_epochs, n_batch, num_batches,
                                    d_error, g_error, d_pred_real, d_pred_fake
                                    )
            
