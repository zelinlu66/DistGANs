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
from utils import *
import time
import PIL.Image as pil
import numpy as np
import os
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod


class GANs_model(object):
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
        if self.verbose:
            print(*args, **kwargs)

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def save_models(self):
        # G_directory = self.createFolder("/G_model")
        # D_directory = self.createFolder("/D_model")
        filename_D = 'D_state_dict.pth'
        filename_G = 'G_state_dict.pth'
        torch.save(self.G.state_dict(), filename_G)
        torch.save(self.D.state_dict(), filename_D)

    def build_models(self):
        D = self.build_discriminator()
        G = self.build_generator()
        return D, G

    @abstractmethod
    def build_discriminator(self):
        pass

    @abstractmethod
    def build_generator(self):
        pass

    def optimizer_initialize(
        self, loss, lr_x, lr_y, optimizer_name='SGD', label_smoothing=False
    ):
        if optimizer_name == 'Jacobi':
            self.optimizer = Jacobi(
                self.G, self.D, loss, lr_x, lr_y, label_smoothing
            )
        elif optimizer_name == 'CGD':
            self.optimizer = CGD(self.G, self.D, loss, lr_x)
        elif optimizer_name == 'Newton':
            self.optimizer = Newton(self.G, self.D, loss, lr_x, lr_y)
        elif optimizer_name == 'JacobiMultiCost':
            self.optimizer = JacobiMultiCost(self.G, self.D, loss, lr_x, lr_y)
        elif optimizer_name == 'GaussSeidel':
            self.optimizer = GaussSeidel(self.G, self.D, loss, lr_x, lr_y)
        else:
            self.optimizer = SGD(self.G, self.D, loss, lr_x)

    def save_images(self, epoch_number, n_batch, images):
        count = 0
        for image_index in range(0, images.shape[0]):
            count = count + 1
            if self.imtype == 'RGB':
                image = images[image_index]  # [0]
                image = image.detach().numpy()
                image = (image + 1) / 2
                image = image.transpose([1, 2, 0])
                self.createFolder(self.save_path)
                path = str(
                    self.save_path
                    + '/fake_image'
                    + '_Epoch_'
                    + str(e + 1)
                    + '_Batch_'
                    + str(n_batch)
                    + '_N_image_'
                    + str(count)
                    + '.png'
                )
                plt.imsave(path, image)
            else:
                image = images[image_index][0]
                image = image.detach().numpy()
                image = (image + 1) / 2
                img = pil.fromarray(np.uint8(image * 255), 'L')
                self.createFolder(self.save_path)
                path = str(
                    self.save_path
                    + '/fake_image'
                    + '_Epoch_'
                    + str(epoch_number + 1)
                    + '_Batch_'
                    + str(n_batch)
                    + '_N_image_'
                    + str(count)
                    + '.png'
                )
                img.save(path)

    @abstractmethod
    def train(
        self,
        loss=torch.nn.BCEWithLogitsLoss(),
        lr_x=torch.tensor([0.001]),
        lr_y=torch.tensor([0.001]),
        optimizer_name='Jacobi',
        num_epochs=1,
        batch_size=100,
        verbose=True,
        save_path='./data_fake',
        label_smoothing=False,
        single_number=None,
    ):
        pass
