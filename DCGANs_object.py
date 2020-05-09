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
import os
import matplotlib.pyplot as plt

class DCGANs_model(object):
    def __init__(self, data):
        self.data = data
        self.data_dimension = self.data[0][0].numpy().shape
        self.D, self.G = self.build_models()
        self.D_error_real_history = []
        self.D_error_fake_history = []
        self.G_error_history = []
        
        if self.data_dimension[0] == 3:
            self.imtype = 'RGB'
        else:
            self.imtype = 'gray'
        
   
        
    def print_verbose(self, *args, **kwargs):
        if self.verbose :
            print(*args, **kwargs)
            
    def build_models(self):
        D = self.build_discriminator()
        G = self.build_generator()
        return D, G
    
    def createFolder(self,directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)
        
    def build_discriminator(self):
        D = DiscriminativeCNN(self.data_dimension[0])
        D.apply(init_weights)
        return D
    
    def build_generator(self, noise_dimension = 100):
        self.noise_dimension = noise_dimension
        G = GenerativeCNN(noise_dimension, self.data_dimension[0])
        G.apply(init_weights)
        return G
    
    def save_models(self):
        G_directory = self.createFolder("/G_model")
        D_directory = self.createFolder("/D_model")
        torch.save(self.G.state_dict(), G_directory)
        torch.save(self.D.state_dict(), D_directory)
        
    
# loss = torch.nn.BCEWithLogitsLoss()
#loss = binary_cross_entropy
    def train(self,loss = torch.nn.BCEWithLogitsLoss(), lr_x = torch.tensor([0.01]), lr_y = torch.tensor([0.01]), optimizer_name = 'Jacobi', num_epochs = 1, batch_size = 100, verbose = True, save_path = './data_fake_DCGANS',label_smoothing = False, single_number = None):
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=100, shuffle=True)
        if single_number is not None:
            self.num_test_samples = 5
            self.data = [i for i in self.data if i[1] == torch.tensor(single_number)]
            self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=100, shuffle=True)
            self.display_progress = 50
        else: 
            self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=100, shuffle=True)
            self.num_test_samples = 10
            self.display_progress = 100
        self.verbose = verbose
        self.save_path = save_path
        self.test_noise = noise(self.num_test_samples, self.noise_dimension)
        if optimizer_name == 'Jacobi':
            optimizer = Jacobi(self.G, self.D, loss, lr_x = torch.tensor([0.01]), lr_y = torch.tensor([0.01]), label_smoothing = label_smoothing)
        elif optimizer_name == 'CGD':
            optimizer = CGD(self.G, self.D, loss, lr_x)
        elif optimizer_name == 'Newton':
            optimizer = Newton(self.G, self.D, loss, lr_x)
        elif optimizer_name == 'JacobiMultiCost':
            optimizer = JacobiMultiCost(self.G, self.D, loss, lr_x = torch.tensor([0.01]), lr_y = torch.tensor([0.01]))
        elif optimizer_name == 'GaussSeidel':
            optimizer = GaussSeidel(self.G, self.D, loss, lr_x)
        else:
            optimizer = SGD(self.G, self.D, loss, lr_x)
  
        start = time.time()
        for e in range(num_epochs):
            self.print_verbose("######################################################")
            for n_batch, (real_batch,_) in enumerate(self.data_loader):
                real_data = Variable((real_batch))
                N = real_batch.size(0)
                optimizer.zero_grad()
                if optimizer_name == 'GaussSeidel':
                    error_real, error_fake, g_error = optimizer.step(real_data,N)
                    self.D = optimizer.D
                    self.G = optimizer.G
                else:
                    error_real, error_fake, g_error, p_x, p_y = optimizer.step(real_data,N)
                
                    index = 0
                    for p in self.G.parameters():
                        p.data.add_(p_x[index: index + p.numel()].reshape(p.shape))
                        index += p.numel()
                    if index != p_x.numel():
                        raise RuntimeError('CG size mismatch')
                    index = 0
                    for p in self.D.parameters():
                        p.data.add_(p_y[index: index + p.numel()].reshape(p.shape))
                        index += p.numel()
                    if index != p_y.numel():
                        raise RuntimeError('CG size mismatch')
            
                self.D_error_real_history.append(error_real)
                self.D_error_fake_history.append(error_fake)
                self.G_error_history.append(g_error)
                
                self.print_verbose('Epoch: ',str(e + 1 ) ,'/',str(num_epochs))
                self.print_verbose('Batch Number: ', str(n_batch + 1))
                self.print_verbose('Error_discriminator__real: ', "{:.5e}".format(error_real), 'Error_discriminator__fake: ', "{:.5e}".format(error_fake),'Error_generator: ', "{:.5e}".format(g_error))
                
                if (n_batch) % self.display_progress == 0:
                    test_images = optimizer.G(self.test_noise)
                    count = 0
                    for image_index in range(0,test_images.shape[0]):
                        count = count + 1
                        if self.imtype == 'RGB':
                            image = test_images[image_index]#[0]
                            image = image.detach().numpy()
                            image = (image + 1)/2
                            image = image.transpose([1, 2, 0])
                            self.createFolder(self.save_path)
                            path = str(self.save_path + '/fake_image'+'_Epoch_'+str(e + 1)+'_Batch_'+str(n_batch)+'_N_image_'+str(count)+'.png')
                            plt.imsave(path, image)
                        else:
                            image = test_images[image_index][0]
                            image = image.detach().numpy()
                            image = (image + 1)/2
                            img = pil.fromarray(np.uint8(image * 255) , 'L')
                            self.createFolder(self.save_path)
                            path = str(self.save_path + '/fake_image'+'_Epoch_'+str(e + 1)+'_Batch_'+str(n_batch)+'_N_image_'+str(count)+'.png')
                            img.save(path)
                            
                            
                            
                        
            self.print_verbose("######################################################")
        end = time.time()
        self.print_verbose('Total Time[s]: ', str( end - start))

