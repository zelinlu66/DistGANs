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


###########################################
##########################################

class myCGD(object):
    def __init__(self, G, D, eps=1e-8, beta2=0.99, lr=1e-3, solve_x = False):
        self.G_params = list(G.parameters())
        self.D_params = list(D.parameters())
        self.lr = lr
        self.square_avgx = None
        self.square_avgy = None
        self.beta2 = beta2
        self.eps = eps
        self.cg_x = None
        self.cg_y = None  
        self.count = 0
        self.old_x = None
        self.old_y = None
        self.solve_x = solve_x
        
    def zero_grad(self):
        zero_grad(self.G_params)
        zero_grad(self.D_params)

    def step(self, loss):
        self.count += 1
        grad_x = autograd.grad(loss, self.G_params, create_graph=True,
                               retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.D_params, create_graph=True,
                               retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        if self.square_avgx is None and self.square_avgy is None:
            self.square_avgx = torch.zeros(grad_x_vec.size(), requires_grad=False)
            self.square_avgy = torch.zeros(grad_y_vec.size(), requires_grad=False)
        self.square_avgx.mul_(self.beta2).addcmul_(1 - self.beta2, grad_x_vec.data, grad_x_vec.data)
        self.square_avgy.mul_(self.beta2).addcmul_(1 - self.beta2, grad_y_vec.data, grad_y_vec.data)

        # Initialization bias correction
        bias_correction2 = 1 - self.beta2 ** self.count

        lr_x = math.sqrt(bias_correction2) * self.lr / self.square_avgx.sqrt().add(self.eps)
        lr_y = math.sqrt(bias_correction2) * self.lr / self.square_avgy.sqrt().add(self.eps)
        scaled_grad_x = torch.mul(lr_x, grad_x_vec).detach()  # lr_x * grad_x
        scaled_grad_y = torch.mul(lr_y, grad_y_vec).detach()  # lr_y * grad_y
        hvp_x_vec = Hvp_vec(grad_y_vec, self.G_params, scaled_grad_y,
                           retain_graph=True)  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.D_params, scaled_grad_x,
                           retain_graph=True)  # D_yx * lr_x * grad_x

        p_x = torch.add(grad_x_vec, - hvp_x_vec).detach_()  # grad_x - D_xy * lr_y * grad_y
        p_y = torch.add(grad_y_vec, hvp_y_vec).detach_()  # grad_y + D_yx * lr_x * grad_x
        
        if self.solve_x:
            p_y.mul_(lr_y.sqrt())
            # p_y_norm = p_y.norm(p=2).detach_()
            # if self.old_y is not None:
            #     self.old_y = self.old_y / p_y_norm
            cg_y, self.iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                             x_params=self.D_params,
                                                             y_params=self.G_params, kk=p_y,
                                                             x=self.old_y,
                                                             nsteps=p_y.shape[0] // 10000,
                                                             lr_x=lr_y, lr_y=lr_x)
            # cg_y.mul_(p_y_norm)
            cg_y.detach_().mul_(- lr_y.sqrt())
            hcg = Hvp_vec(grad_y_vec, self.G_params, cg_y, retain_graph=True).add_(
                grad_x_vec).detach_()
            # grad_x + D_xy * delta y
            cg_x = hcg.mul(lr_x)
            self.old_x = hcg.mul(lr_x.sqrt())
        else:

            p_x.mul_(lr_x.sqrt())
            # p_x_norm = p_x.norm(p=2).detach_()
            # if self.old_x is not None:
            #     self.old_x = self.old_x / p_x_norm
            cg_x, self.iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                             x_params=self.G_params,
                                                             y_params=self.D_params, kk=p_x,
                                                             x=self.old_x,
                                                             nsteps=p_x.shape[0] // 10000,
                                                             lr_x=lr_x, lr_y=lr_y)
            # cg_x.detach_().mul_(p_x_norm)
            cg_x.detach_().mul_(lr_x.sqrt())  # delta x = lr_x.sqrt() * cg_x
            hcg = Hvp_vec(grad_x_vec, self.D_params, cg_x, retain_graph=True).add_(grad_y_vec).detach_()
            # grad_y + D_yx * delta x
            cg_y = hcg.mul(- lr_y)
            self.old_y = hcg.mul(lr_y.sqrt())
            
         
        index = 0
        for p in self.G_params:
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in self.D_params:
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise RuntimeError('CG size mismatch')
        
        self.solve_x = False if self.solve_x else True
#########################################################
class myCGD_fg(object):
    def __init__(self, G, D, eps=1e-8, beta2=0.99, lr=1e-3, solve_x = False):
        self.G_params = list(G.parameters())
        self.D_params = list(D.parameters())
        self.lr = lr
        self.square_avgx = None
        self.square_avgy = None
        self.beta2 = beta2
        self.eps = eps
        self.cg_x = None
        self.cg_y = None  
        self.count = 0
        self.old_x = None
        self.old_y = None
        self.solve_x = solve_x
        
    def zero_grad(self):
        zero_grad(self.G_params)
        zero_grad(self.D_params)

    def step(self, f,g):
        self.count += 1
        # Derivatives of f
        grad_f_x = autograd.grad(f, self.G_params, create_graph=True,
                               retain_graph=True)
        grad_f_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_f_x])
        grad_f_y = autograd.grad(f, self.D_params, create_graph=True,
                               retain_graph=True)
        grad_f_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_f_y])
        # Derivatives of g
        grad_g_y = autograd.grad(g, self.D_params, create_graph=True,
                               retain_graph=True)
        grad_g_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_g_y])
        
        grad_g_x = autograd.grad(g, self.G_params, create_graph=True,
                               retain_graph=True)
        grad_g_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_g_x])
        
        if self.square_avgx is None and self.square_avgy is None:
            self.square_avgx = torch.zeros(grad_f_x_vec.size(), requires_grad=False)
            self.square_avgy = torch.zeros(grad_g_y_vec.size(), requires_grad=False)
        self.square_avgx.mul_(self.beta2).addcmul_(1 - self.beta2, grad_f_x_vec.data, grad_f_x_vec.data)
        self.square_avgy.mul_(self.beta2).addcmul_(1 - self.beta2, grad_g_y_vec.data, grad_g_y_vec.data)

        # Initialization bias correction
        bias_correction2 = 1 - self.beta2 ** self.count

        lr_x = math.sqrt(bias_correction2) * self.lr / self.square_avgx.sqrt().add(self.eps)
        lr_y = math.sqrt(bias_correction2) * self.lr / self.square_avgy.sqrt().add(self.eps)
        scaled_grad_f_x = torch.mul(lr_x, grad_f_x_vec).detach()  # lr_x * grad_f_x
        scaled_grad_g_y = torch.mul(lr_y, grad_g_y_vec).detach()  # lr_y * grad_g_y
        # Hessians computations
        
        hvp_x_vec = Hvp_vec(grad_f_y_vec, self.G_params, scaled_grad_g_y,
                           retain_graph=True)  # Df_xy * lr_y * grad_g_y
        hvp_y_vec = Hvp_vec(grad_g_x_vec, self.D_params, scaled_grad_f_x,
                           retain_graph=True)  # Dg_yx * lr_x * grad_f_x

        p_x = torch.add(grad_f_x_vec, - hvp_x_vec).detach_()  # grad_f_x - Df_xy * lr_y * grad_y
        p_y = torch.add(grad_g_y_vec, hvp_y_vec).detach_()  # grad_g_y + Dg_yx * lr_x * grad_x
        
        if self.solve_x:
            p_y.mul_(lr_y.sqrt())
            # p_y_norm = p_y.norm(p=2).detach_()
            # if self.old_y is not None:                             
            #     self.old_y = self.old_y / p_y_norm
            cg_y, self.iter_num = general_conjugate_gradient(grad_x=grad_f_y_vec, grad_y=grad_g_x_vec,
                                                             x_params=self.D_params,
                                                             y_params=self.G_params, kk=p_y,
                                                             x=self.old_y,
                                                             nsteps=p_y.shape[0] // 10000,
                                                             lr_x=lr_y, lr_y=lr_x)
            # cg_y.mul_(p_y_norm)
            cg_y.detach_().mul_(- lr_y.sqrt())
            hcg = Hvp_vec(grad_y_vec, self.G_params, cg_y, retain_graph=True).add_(
                grad_x_vec).detach_()
            # grad_x + D_xy * delta y
            cg_x = hcg.mul(lr_x)
            self.old_x = hcg.mul(lr_x.sqrt())
        else:

            p_x.mul_(lr_x.sqrt())
            # p_x_norm = p_x.norm(p=2).detach_()     (grad_g_x,  grad_f_y ,G, D)
            # if self.old_x is not None:
            #     self.old_x = self.old_x / p_x_norm
            cg_x, self.iter_num = general_conjugate_gradient(grad_g_x_vec, grad_f_y_vec,
                                                             x_params=self.G_params,
                                                             y_params=self.D_params, kk=p_x,
                                                             x=self.old_x,
                                                             nsteps=p_x.shape[0] // 10000,
                                                             lr_x=lr_x, lr_y=lr_y)
            # cg_x.detach_().mul_(p_x_norm)
            cg_x.detach_().mul_(lr_x.sqrt())  # delta x = lr_x.sqrt() * cg_x
            hcg = Hvp_vec(grad_x_vec, self.D_params, cg_x, retain_graph=True).add_(grad_y_vec).detach_()
            # grad_y + D_yx * delta x
            cg_y = hcg.mul(- lr_y)
            self.old_y = hcg.mul(lr_y.sqrt())
            
         
        index = 0
        for p in self.G_params:
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in self.D_params:
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise RuntimeError('CG size mismatch')
        
        self.solve_x = False if self.solve_x else True
######################################
class myCGD_Jacobi(object):
    def __init__(self, G, D, lr=1e-3):
        self.G_params = list(G.parameters())
        self.D_params = list(D.parameters())
        self.lr = lr
        self.count = 0
        
    def zero_grad(self):
        zero_grad(self.G_params)
        zero_grad(self.D_params)

    def step(self, loss):
        self.count += 1
        grad_x = autograd.grad(loss, self.G_params, create_graph=True,
                               retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.D_params, create_graph=True,
                               retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        hvp_x_vec = Hvp_vec(grad_y_vec, self.G_params, grad_y_vec,
                           retain_graph=True)  # D_xy * grad_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.D_params, grad_x_vec,
                           retain_graph=True)  # D_yx * grad_x
        

        p_x = torch.add(grad_x_vec, 2*hvp_x_vec).detach_()  # grad_x  +2 * D_xy * grad_y
        p_y = torch.add(grad_y_vec, 2*hvp_y_vec).detach_()  # grad_y  +2 * D_yx * grad_x
        
        update_x = torch.mul(self.lr,p_x)
        update_y = torch.mul(-self.lr,p_y)
         
        index = 0
        for p in self.G_params:
            p.data.add_(update_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != update_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in self.D_params:
            p.data.add_(update_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != update_y.numel():
            raise RuntimeError('CG size mismatch')
        