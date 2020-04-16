# -*- coding: utf-8 -*-
"""
@authors: Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)

"""

from GANs_object import *
from DCGANs_object import *
#model = GANs_model(mnist_data())
#model.train(num_epochs = 15, lr = torch.tensor([0.05]), optimizer = 'Newton', verbose = True, save_path = 'WGAN') # save_path = './J_BCE'
model = DCGANs_model(cifar_data_dcgans())
model.train(num_epochs = 1, lr = torch.tensor([0.01]), optimizer = 'Jacobi', verbose = True) #save_path = 'WGAN') # save_path = './J_BCE'