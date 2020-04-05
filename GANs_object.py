# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:34:40 2020

@authors: Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)

"""
import torch
import numpy
from models import *
from optimizers import *
from Dataloader import *
import time
import PIL.Image as pil
import numpy as np

class GANs_model(object):
    def __init__(self, data):
        self.data = data
        self.data_dimension = self.data[0][0].numpy().shape
        self.D, self.G = self.build_models()
        self.D_error_real_history = []
        self.D_error_fake_history = []
        self.G_error_history = []
        
   
        
    def print_verbose(self, *args, **kwargs):
        if self.verbose :
            print(*args, **kwargs)
            
    def build_models(self):
        D = self.build_discriminator()
        G = self.build_generator()
        return D, G
        
    def build_discriminator(self):
        n_features = numpy.prod(self.data_dimension)
        D = Discriminator(n_features)
        return D
    
    def build_generator(self, noise_dimension = 100):
        self.noise_dimension = noise_dimension
        n_out = numpy.prod(self.data_dimension)
        G = Generator(noise_dimension, n_out)
        return G
    
    
    def train(self,loss = torch.nn.BCEWithLogitsLoss(), lr = torch.tensor([0.0001]), optimizer = 'Jacobi', num_epochs = 1, batch_size = 100, verbose = True):
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=100, shuffle=True)
        self.verbose = verbose
        self.num_test_samples = 16
        self.test_noise = noise(self.num_test_samples, self.noise_dimension)
        if optimizer == 'Jacobi':
            optimizer = Jacobi(self.G, self.D, loss, lr)
        elif optimizer == 'CGD':
            optimizer = CGD(self.G, self.D, loss, lr)
        else:
            optimizer = SGD(self.G, self.D, loss, lr)
  
        start = time.time()
        for e in range(num_epochs):
            self.print_verbose("######################################################")
            for n_batch, (real_batch,_) in enumerate(self.data_loader):
                N = real_batch.size(0)
                real_data = Variable(images_to_vectors(real_batch))
                fake_data = self.G(noise(N, 100)) # Second argument of noise is the noise_dimension parameter of build_generator
                optimizer.zero_grad()
                error_real, error_fake, g_error = optimizer.step(real_data,fake_data,N) 
                self.D_error_real_history.append(error_real)
                self.D_error_fake_history.append(error_fake)
                self.G_error_history.append(g_error)
                
                self.print_verbose('Epoch: ',str(e + 1 ) ,'/',str(num_epochs))
                self.print_verbose('Batch Number: ', str(n_batch + 1))
                self.print_verbose('Error_discriminator__real: ', round(error_real.item(),3), 'Error_discriminator__fake: ', round(error_fake.item(),3),'Error_generator: ', round(g_error.item(),3))
                
                if (n_batch) % 100 == 0: 
                    test_images = vectors_to_images(self.G(self.test_noise), self.data_dimension) # data_dimension: dimension of output image ex: [1,28,28]
                    count = 0
                    for image_index in range(0,test_images.shape[0]):
                        count = count + 1
                        image = test_images[image_index][0]
                        image = image.detach().numpy()
                        image = (image + 1)/2
                        img = pil.fromarray(np.uint8(image * 255) , 'L')
                        img.save('./data_mnist/fake_image'+'_Epoch_'+str(e)+'_Batch_'+str(n_batch)+'_N_image_'+str(count)+'.png')
                        
            self.print_verbose("######################################################")
        end = time.time()
        self.print_verbose('Total Time[s]: ', str( end - start))

