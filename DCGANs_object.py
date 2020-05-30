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
import GANs_abstract_object
from GANs_abstract_object import *


class DCGANs_model(GANs_model):
    def __init__(self, data):
        super(DCGANs_model, self).__init__(data)

    def build_discriminator(self):
        D = DiscriminatorCNN(self.data_dimension[0], self.data_dimension[1])
        D.apply(weights_init_normal)
        return D

    def build_generator(self, noise_dimension=100):
        self.noise_dimension = noise_dimension
        G = GeneratorCNN(
            noise_dimension, self.data_dimension[0], self.data_dimension[1]
        )
        G.apply(weights_init_normal)
        return G

    # loss = torch.nn.BCEWithLogitsLoss()
    # loss = binary_cross_entropy
    def train(
        self,
        loss=torch.nn.BCEWithLogitsLoss(),
        lr_x=torch.tensor([0.01]),
        lr_y=torch.tensor([0.01]),
        optimizer_name="Jacobi",
        num_epochs=1,
        batch_size=100,
        verbose=True,
        save_path="./data_fake_DCGANS",
        label_smoothing=False,
        single_number=None,
    ):
        self.data_loader = torch.utils.data.DataLoader(
            self.data, batch_size=100, shuffle=True
        )
        if single_number is not None:
            self.num_test_samples = 5
            self.data = [i for i in self.data if i[1] == torch.tensor(single_number)]
            self.data_loader = torch.utils.data.DataLoader(
                self.data, batch_size=100, shuffle=True
            )
            self.display_progress = 50
        else:
            self.data_loader = torch.utils.data.DataLoader(
                self.data, batch_size=100, shuffle=True
            )
            self.num_test_samples = 10
            self.display_progress = 100

        self.verbose = verbose
        self.save_path = save_path
        self.test_noise = noise(self.num_test_samples, self.noise_dimension)
        self.optimizer_initialize(loss, lr_x, lr_y, optimizer_name, label_smoothing)
        start = time.time()
        for e in range(num_epochs):
            self.print_verbose("######################################################")
            for n_batch, (real_batch, _) in enumerate(self.data_loader):
                real_data = Variable((real_batch))
                N = real_batch.size(0)
                self.optimizer.zero_grad()
                if optimizer_name == "GaussSeidel":
                    error_real, error_fake, g_error = self.optimizer.step(real_data, N)
                    self.D = optimizer.D
                    self.G = optimizer.G
                else:
                    (error_real, error_fake, g_error, p_x, p_y,) = self.optimizer.step(
                        real_data, N
                    )

                    index = 0
                    for p in self.G.parameters():
                        p.data.add_(p_x[index : index + p.numel()].reshape(p.shape))
                        index += p.numel()
                    if index != p_x.numel():
                        raise RuntimeError("CG size mismatch")
                    index = 0
                    for p in self.D.parameters():
                        p.data.add_(p_y[index : index + p.numel()].reshape(p.shape))
                        index += p.numel()
                    if index != p_y.numel():
                        raise RuntimeError("CG size mismatch")

                self.D_error_real_history.append(error_real)
                self.D_error_fake_history.append(error_fake)
                self.G_error_history.append(g_error)

                self.print_verbose("Epoch: ", str(e + 1), "/", str(num_epochs))
                self.print_verbose("Batch Number: ", str(n_batch + 1))
                self.print_verbose(
                    "Error_discriminator__real: ",
                    "{:.5e}".format(error_real),
                    "Error_discriminator__fake: ",
                    "{:.5e}".format(error_fake),
                    "Error_generator: ",
                    "{:.5e}".format(g_error),
                )

                if (n_batch) % self.display_progress == 0:
                    test_images = self.optimizer.G(self.test_noise)
                    self.save_images(e, n_batch, test_images)

            self.print_verbose("######################################################")
        end = time.time()
        self.print_verbose("Total Time[s]: ", str(end - start))
