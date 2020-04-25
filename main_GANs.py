# -*- coding: utf-8 -*-
"""
@authors: Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)

"""

from GANs_object import *
from DCGANs_object import *


'''
! READ ME !

X = Generator
Y = Discriminaror

BCEWithLogitsLoss is the default cost function, to change it to standard BCE the code needs to be changed
in line 68 of GANs_object or line 67 of DCGANs_object

Different learning rates for X and Y can only be used with 'Jacobi' and 'JacobiMultiCost'
for the other optimizers the learning rate will be set to the value of lr_x

Label smoothing variation is implemented only for optimizer 'Jacobi' and only for GANs_object

To swich from GANs (DCGANs) to DCGANs (GANs) comment lines 35,36,37 (40,41,42) and uncomment 40,41,42 (35,36,37)

TO TRY: Setting much lower learning rates to see if model collapse is avoided (ex. lr_x = 0.0001 , lr_y = 0.0004)

'''


#model = GANs_model(mnist_data())
#model.train(num_epochs = 1, lr_x = torch.tensor([0.01]), lr_y = torch.tensor([0.01]), 
#            optimizer = 'Jacobi', verbose = True, label_smoothing = False) # save_path = ''


model = DCGANs_model(mnist_data_dcgans())
model.train(num_epochs = 1, lr_x = torch.tensor([0.01]), lr_y = torch.tensor([0.01]), 
           optimizer = 'Jacobi', verbose = True, label_smoothing = False) #save_path = ''

plt.figure()
plt.plot([x for x in range(0,len(model.D_error_real_history))], model.D_error_real_history)
plt.plot([x for x in range(0,len(model.D_error_fake_history))], model.D_error_fake_history)
plt.plot([x for x in range(0,len(model.G_error_history))], model.G_error_history)
plt.xlabel('Number of epochs')
plt.ylabel('Loss function value')
plt.legend(['Discriminator: Loss on Real Data', 'Discriminator: Loss on Fake Data', 'Generator: Loss'])
plt.savefig('cost_report.png')