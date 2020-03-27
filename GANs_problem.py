import torch
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from foo import *
from torch import autograd
from optimizers import *
from Dataloader import *
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.models.inception import inception_v3

DIM = 64

def transform(x):
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5


# Create discriminator network


###############################################################################

def ones_target(size1, size2):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size1, size2))
    return data

def zeros_target(size1,size2):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size1, size2))
    return data



learning_rate = 0.0001
lr = learning_rate
batch_size = 64
z_dim = 128
dropout = None

discriminator = GoodDiscriminator()
generator = GoodGenerator()

#dataset = CIFAR10(root='datas/cifar10', download=True, train=True, transform=transform)
#dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,drop_last=True)


loss = nn.BCEWithLogitsLoss()
#noise_shape=(64, z_dim)
first_noise = torch.randn(2,128)

params_G = generator.state_dict()
params_D = discriminator.state_dict()

old_gen = GoodGenerator()
old_gen.load_state_dict(params_G)

prevx = old_gen(first_noise) # prev_x
prevx_y = discriminator(prevx) # noise---->old_gen---->discriminator

x = generator(first_noise) #x

old_dis = GoodDiscriminator()
old_dis.load_state_dict(params_D, strict=False)

x_prevy = old_dis(x)  # noise---->gen---->old_discriminator

# f = loss_dis g = loss_gen
# x = gen   Y = dis

loss_dis_prevdis = loss(x_prevy, zeros_target(2,1)) #f(x,prev_y)
loss_dis_prevgen = loss(prevx_y, zeros_target(2,1)) #f(prev_x, y)
loss_gen_prevdis = loss(x_prevy, ones_target(2,1)) #g(x,prev_y)
loss_gen_prevgen = loss(prevx_y, ones_target(2,1)) #g(prev_x, y)

f_val_x = loss_dis_prevdis
f_val_y = loss_dis_prevgen
g_val_x = loss_gen_prevdis
g_val_y = loss_gen_prevgen

grad_f_x_x  = autograd.grad(f_val_x, generator.parameters(), create_graph=True, allow_unused=True)
grad_f_x_y  = autograd.grad(f_val_y, old_gen.parameters(), create_graph=True, allow_unused=True)
grad_g_y_x  = autograd.grad(g_val_x, old_dis.parameters(), create_graph=True, allow_unused=True)
grad_g_y_y  = autograd.grad(g_val_y, discriminator.parameters(), create_graph=True, allow_unused=True)

grad_f_x_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_f_x_x])
grad_f_x_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_f_x_y])
grad_g_y_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_g_y_x])
grad_g_y_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_g_y_y])

hess_f_xy_x = Hvp_vec(grad_f_x_x_vec, old_dis.parameters(), grad_g_y_x_vec, retain_graph=True)  # h_xy * d_y

#hess_f_xy_x = autograd.grad(grad_f_x_x_vec, old_dis.parameters(), grad_outputs = grad_g_y_x_vec, allow_unused = True, retain_graph = True)
#hess_f_xy_y = autograd.grad(grad_f_x_y_vec, discriminator.parameters(), grad_outputs = torch.ones_like(grad_f_x_y_vec), allow_unused = True, retain_graph = True)
#hess_g_yx_x = autograd.grad(grad_g_y_x_vec, generator.parameters(), grad_outputs = torch.ones_like(grad_g_y_x_vec), allow_unused = True, retain_graph = True)
hess_g_yx_y = autograd.grad(grad_g_y_y_vec, old_gen.parameters(), grad_outputs = torch.ones_like(grad_g_y_y_vec), allow_unused = True, retain_graph = True)

#delta_x = -lr * (grad_f_x_x + 2* hess_f_xy_x * grad_g_y_x) 
#delta_y = -lr * (grad_g_y_y + 2* hess_g_yx_y * grad_f_x_y)

'''
#[f_history_cgd_jacobi, g_history_cgd_jacobi, x_history_cgd_jacobi, y_history_cgd_jacobi] = competitive_cgd_jacobi_solver.solve(f, g, x_cgd_jacobi, y_cgd_jacobi)

def competitive_gradient_jacobi(f, g, x0, y0, nsteps=10, residual_tol=1e-10, lr=1e-3, verbose=True, hessian_vec=True, delay=1):
    iter_count = 0
    f_history = []
    g_history = []
    x_history = []
    y_history = []    
    x = x0
    y = y0
    prev_y = y.clone().detach().requires_grad_(True)
    prev_x = x.clone().detach().requires_grad_(True)
    x_history.append(x0)
    y_history.append(y0)
    x_buffer = []
    y_buffer = []
    while iter_count < nsteps:
        iter_count += 1
        f_val_x = f(x,prev_y)
        f_val_y = f(prev_x, y)
        g_val_x = g(x,prev_y)
        g_val_y = g(prev_x, y)
        grad_f_x_x, = autograd.grad(f_val_x, x, create_graph=True, allow_unused=True) # terrible variable name, implies diagonal hessian!!
        grad_f_x_y, = autograd.grad(f_val_y, prev_x, create_graph=True, allow_unused=True) # terrible variable name, implies diagonal hessian!!
        grad_g_y_x, = autograd.grad(g_val_x, prev_y, create_graph=True, allow_unused=True)
        grad_g_y_y, = autograd.grad(g_val_y, y, create_graph=True, allow_unused=True)
        if hessian_vec:
            hess_f_xy_x = Hessian_vec(grad_f_x_x, prev_y, retain_graph=False)
            hess_f_xy_y = Hessian_vec(grad_f_x_y, y, retain_graph=False)
            hess_g_yx_x = Hessian_vec(grad_g_y_x, x, retain_graph=False)
            hess_g_yx_y = Hessian_vec(grad_g_y_y, prev_x, retain_graph=False)
            delta_x = -lr * (grad_f_x_x + 2* hess_f_xy_x * grad_g_y_x) 
            delta_y = -lr * (grad_g_y_y + 2* hess_g_yx_y * grad_f_x_y)
        else:
            hess_f_xy_x = Hessian(grad_f_x_x, prev_y, retain_graph=False)
            hess_f_xy_y = Hessian(grad_f_x_y, y, retain_graph=False)
            hess_g_yx_x = Hessian(grad_g_y_x, x, retain_graph=False)
            hess_g_yx_y = Hessian(grad_g_y_y, prev_x, retain_graph=False)
            delta_x = -lr * (grad_f_x_x  + 2* hess_f_xy_x @ grad_g_y_x) 
            delta_y = -lr * (grad_g_y_y  + 2* hess_g_yx_y @ grad_f_x_y)

        new_x = x - lr * delta_x
        new_y = y - lr * delta_y
        x = new_x.clone().detach().requires_grad_(True)
        y = new_y.clone().detach().requires_grad_(True)
        # if iter_count > 1:
        x_buffer.append(x)
        y_buffer.append(y)
        if verbose == 1:
            print("######################################################")
            print("Iteration: ", iter_count)
            print("x: ", x)
            print("y: ", y)
            print("f(x,y): ", f(x,y))
            print("g(x,y): ", g(x,y))
            print("hess_f_xy_x:", hess_f_xy_x)
            print("hess_f_xy_y:", hess_f_xy_y)
            print("hess_g_yx_x:", hess_g_yx_x)
            print("hess_g_yx_y:", hess_g_yx_y)
            print("######################################################")
                  
        
        f_history.append(f(x,y))
        g_history.append(g(x,y))
        x_history.append(x)
        y_history.append(y)
        if iter_count > delay:
            prev_y = y_buffer[iter_count - delay].clone().detach().requires_grad_(True)
            prev_x = x_buffer[iter_count - delay].clone().detach().requires_grad_(True)

               
    return f_history, g_history, x_history, y_history  






'''
































'''
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()



def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


num_test_samples = 16
test_noise = noise(num_test_samples)

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')
# Total number of epochs to train
num_epochs = 1
for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = \
              train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
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
    '''