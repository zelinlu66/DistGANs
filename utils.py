'''
@authors: Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)
'''
##########################################
import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import autograd
from torch.autograd.variable import Variable
import math

# Utils specific for MNIST
# Utils for Hessian computation and matrix inverse
# Logger


################################################################

''' UTILS PER IL MY MNIST'''
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def noise(size, noise_size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, noise_size))
    return n

def images_to_vectors(images):
    image_dim = images.size(1)*images.size(2)*images.size(3)
    return images.view(images.size(0), image_dim) 

def vectors_to_images(vectors, array_dim):
    return vectors.view(vectors.size(0), array_dim[0],array_dim[1],array_dim[2])

def images_to_vectors_cifar10(images):
    return images.view(images.size(0), 3072)

def vectors_to_images_cifar10(vectors):
    return vectors.view(vectors.size(0), 3, 32, 32)

#############################################################################
    
def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()
    
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)
            

def Hvp_vec(grad_vec, params, vec, retain_graph=False):
    if torch.isnan(grad_vec).any():
        print('grad vec nan')
        raise ValueError('grad Nan')
    if torch.isnan(vec).any():
        print('vec nan')
        raise ValueError('vec Nan')
    try:
        grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec, retain_graph=retain_graph)
        hvp = torch.cat([g.contiguous().view(-1) for g in grad_grad])
        if torch.isnan(hvp).any():
            print('hvp nan')
            raise ValueError('hvp Nan')
    except:
        # print('filling zero for None')
        grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec, retain_graph=retain_graph,
                                  allow_unused=True)
        grad_list = []
        for i, p in enumerate(params):
            if grad_grad[i] is None:
                grad_list.append(torch.zeros_like(p))
            else:
                grad_list.append(grad_grad[i].contiguous().view(-1))
        hvp = torch.cat(grad_list)
        if torch.isnan(hvp).any():
            raise ValueError('hvp Nan')
    return hvp


def general_conjugate_gradient(grad_x, grad_y, x_params, y_params, kk, lr_x, lr_y, x=None, nsteps=10,
                               residual_tol=1e-16,
                               device=torch.device('cpu')):
    '''

    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b:
    :param lr_x:
    :param lr_y:      
    :param x:            
    :param nsteps:
    :param residual_tol:
    :param device:
    :return: (I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) ** -1 * b

    '''
    if x is None:
        x = torch.zeros(kk.shape[0], device=device)
    if grad_x.shape != kk.shape:
        raise RuntimeError('CG: hessian vector product shape mismatch')
    lr_x = lr_x.sqrt()
    mm = kk.clone().detach()
    jj = mm.clone().detach()
    rdotr = torch.dot(mm, mm)
    residual_tol = residual_tol * rdotr
    x_params = tuple(x_params)
    y_params = tuple(y_params)
    for i in range(nsteps):
        # To compute Avp
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True)
        h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * jj, retain_graph=True).mul_(lr_y)
        # h_1.mul_(lr_y)
        # lr_y * D_yx * b
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=lr_y * h_1, retain_graph=True)
        h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).mul_(lr_x)
        # h_2.mul_(lr_x)
        # lr_x * D_xy * lr_y * D_yx * b
        Avp_ = jj + h_2

        alpha = rdotr / torch.dot(jj, Avp_)
        x.data.add_(alpha * jj)
        mm.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(mm, mm)
        beta = new_rdotr / rdotr
        jj = mm + beta * jj
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x, i + 1

#######################################################################
def general_conjugate_gradient_jacobi(grad_x, x_params, right_side, lr, x=None, nsteps=10,
                               residual_tol=1e-16,
                               device=torch.device('cpu')):
    '''

    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b:
    :param lr_x:
    :param lr_y:      
    :param x:            
    :param nsteps:
    :param residual_tol:
    :param device:
    :return: (A) ** -1 * (right_side)

    '''
    if x is None:
        x = torch.zeros(right_side.shape[0], device=device)
    lr = lr.sqrt()
    
    right_side_clone1 = right_side.clone().detach()
    right_side_clone2 = right_side_clone1.clone().detach()
    
    rdotr = torch.dot(right_side_clone1, right_side_clone1)
    residual_tol = residual_tol * rdotr
    x_params = tuple(x_params)
    
    for i in range(nsteps):
        h_1 = Hvp_vec(grad_vec=grad_x, params=x_params, vec=lr * x, retain_graph=True).mul_(lr)
        H = -2*h_1 + x
        Avp_ = right_side_clone2 + H

        alpha = rdotr / torch.dot(right_side_clone2, Avp_)
        x.data.add_(alpha * right_side_clone2)
        right_side_clone1.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(right_side_clone1, right_side_clone1)
        beta = new_rdotr / rdotr
        right_side_clone2 = right_side_clone1 + beta * right_side_clone2
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x, i + 1

###########################################
