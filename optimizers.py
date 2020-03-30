# This is a preliminary version of the code
from typing import Any
import time
import torch
import numpy
from torch import Tensor
from torch import autograd
from torch.autograd import Variable
from torch.autograd import grad
from foo import *

def hessian_vec(grad_vec, var, retain_graph=False):
    v = torch.ones_like(var)
    vec, = autograd.grad(grad_vec, var, grad_outputs=v, allow_unused=True, retain_graph=retain_graph)
    return vec

def hessian(grad_vec, var, retain_graph=False):
    v = torch.eye(var.shape[0])
    matrix = torch.cat([autograd.grad(grad_vec, var, grad_outputs=v_row, allow_unused=True, retain_graph=retain_graph)[0]
                            for v_row in v])
    matrix = matrix.view(-1,var.shape[0])
    return matrix
'''
ACGD: Original implementation by Shafer
myCGD: Ours implementation of CGD
myCGDJacobi: Ours implementation of CGD with Jacoby mehod
myCGD_fg: Ours implementation of CGD with 2 different cost functions( NOT READY)
'''
class Richardson(object):

    def __init__(self, matrix, rhs, tol, maxiter, relaxation, verbose=False):

        """
        :param matrix: coefficient matrix
        :param rhs: right hand side
        :param tol: tolerance for stopping criterion based on the relative residual
        :param maxiter: maximum number of iterations
        :param relaxation: relaxation parameter for Richardson
        :param initial_guess: initial guess
        :return: matrix ** -1 * rhs
        """

        self.rhs = rhs
        self.matrix = matrix
        self.tol = tol
        self.maxiter = maxiter
        self.relaxation = relaxation
        self.rhs_norm = torch.norm(rhs, 2)
        self.iteration_count = 0
        self.verbose = verbose

    def print_verbose(self, *args, **kwargs):
        if self.verbose :
            print(*args, **kwargs)

    def solve(self, initial_guess):
        ## TODO: consider passing initial guess to solve()

        residual = self.rhs - self.matrix @ initial_guess
        residual_norm = residual.norm()
        relative_residual_norm = residual_norm / self.rhs_norm

        solution = initial_guess

        while relative_residual_norm > self.tol and self.iteration_count < self.maxiter:
            ## TODO: consider making all of these non-attributes and just return them
            solution = solution + self.relaxation * residual
            
            residual = self.rhs - torch.matmul(self.matrix, solution)
            residual_norm = residual.norm()
            relative_residual_norm = residual_norm / self.rhs_norm
            self.iteration_count += 1
            self.print_verbose("Richardson converged in ", str(self.iteration_count), " iteration with relative residual norm: ",
                                     str(relative_residual_norm), end='...')

        # Do not return because it's already an attribute
        return solution


class ACGD(object):  # Support multi GPU
    def __init__(self, max_params, min_params, eps=1e-8, beta2=0.99, lr=1e-3,
                 solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.lr = lr
        self.solve_x = solve_x
        self.collect_info = collect_info
        self.square_avgx = None
        self.square_avgy = None
        self.beta2 = beta2
        self.eps = eps
        self.cg_x = None
        self.cg_y = None
        self.count = 0

        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, self.iter_num
        else:
            raise ValueError(
                'No update information stored. Set get_norms True before call this method')

    def step(self, loss):
        self.count += 1
        grad_x = autograd.grad(loss, self.max_params, create_graph=True,retain_graph=True, allow_unused = True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True,
                               retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        if self.square_avgx is None and self.square_avgy is None:
            self.square_avgx = torch.zeros(grad_x_vec.size(), requires_grad=False,
                                           )
            self.square_avgy = torch.zeros(grad_y_vec.size(), requires_grad=False,
                                           )
        self.square_avgx.mul_(self.beta2).addcmul_(1 - self.beta2, grad_x_vec.data, grad_x_vec.data)
        self.square_avgy.mul_(self.beta2).addcmul_(1 - self.beta2, grad_y_vec.data, grad_y_vec.data)

        # Initialization bias correction
        bias_correction2 = 1 - self.beta2 ** self.count

        lr_x = math.sqrt(bias_correction2) * self.lr / self.square_avgx.sqrt().add(self.eps)
        lr_y = math.sqrt(bias_correction2) * self.lr / self.square_avgy.sqrt().add(self.eps)
        scaled_grad_x = torch.mul(lr_x, grad_x_vec).detach()  # lr_x * grad_x
        scaled_grad_y = torch.mul(lr_y, grad_y_vec).detach()  # lr_y * grad_y
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, scaled_grad_y,
                           retain_graph=True)  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, scaled_grad_x,
                           retain_graph=True)  # D_yx * lr_x * grad_x

        p_x = torch.add(grad_x_vec, - hvp_x_vec).detach_()  # grad_x - D_xy * lr_y * grad_y
        p_y = torch.add(grad_y_vec, hvp_y_vec).detach_()  # grad_y + D_yx * lr_x * grad_x

        if self.collect_info:
            self.norm_px = lr_x.max()
            self.norm_py = lr_y.max()
            self.timer = time.time()
        if self.solve_x:
            p_y.mul_(lr_y.sqrt())
            # p_y_norm = p_y.norm(p=2).detach_()
            # if self.old_y is not None:
            #     self.old_y = self.old_y / p_y_norm
            cg_y, self.iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                             x_params=self.min_params,
                                                             y_params=self.max_params, b=p_y,
                                                             x=self.old_y,
                                                             nsteps=p_y.shape[0] // 10000,
                                                             lr_x=lr_y, lr_y=lr_x,
                                                             )
            # cg_y.mul_(p_y_norm)
            cg_y.detach_().mul_(- lr_y.sqrt())
            hcg = Hvp_vec(grad_y_vec, self.max_params, cg_y, retain_graph=True).add_(
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
                                                             x_params=self.max_params,
                                                             y_params=self.min_params, b=p_x,
                                                             x=self.old_x,
                                                             nsteps=p_x.shape[0] // 10000,
                                                             lr_x=lr_x, lr_y=lr_y,
                                                             )
            # cg_x.detach_().mul_(p_x_norm)
            cg_x.detach_().mul_(lr_x.sqrt())  # delta x = lr_x.sqrt() * cg_x
            hcg = Hvp_vec(grad_x_vec, self.min_params, cg_x, retain_graph=True).add_(
                grad_y_vec).detach_()
            # grad_y + D_yx * delta x
            cg_y = hcg.mul(- lr_y)
            self.old_y = hcg.mul(lr_y.sqrt())

        if self.collect_info:
            self.timer = time.time() - self.timer

        index = 0
        for p in self.max_params:
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in self.min_params:
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise RuntimeError('CG size mismatch')
        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)

        self.solve_x = False if self.solve_x else True


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
# CGD_fg NOT READY YET !!
'''
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
'''
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
        scaled_grad_x = torch.mul(self.lr,grad_x_vec)
        scaled_grad_y = torch.mul(self.lr,grad_y_vec)

        hvp_x_vec = Hvp_vec(grad_y_vec, self.G_params, scaled_grad_y,
                           retain_graph=True)  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.D_params, scaled_grad_x,
                           retain_graph=True)  # D_yx * lr_x * grad_x
        
        hvp_x_vec = Hvp_vec(grad_y_vec, self.G_params, torch.cat([param.view(-1) for param in self.D_params]),retain_graph=True)  # D_xy * lr_y * y 
        hvp_y_vec = Hvp_vec(grad_x_vec, self.D_params, torch.cat([param.view(-1) for param in self.G_params]),retain_graph=True)  # D_yx * lr_x * x
        

        p_x = torch.add(- grad_x_vec, - hvp_x_vec).detach_()  # grad_x + D_xy * lr_y * y
        p_y = torch.add(grad_y_vec, hvp_y_vec).detach_()  # grad_y + D_yx * lr_x * x
        p_x.mul_(self.lr.sqrt())
        p_y.mul_(self.lr.sqrt())
         
        index = 0
        for p in self.G_params:
            p.data.add_(p_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != p_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in self.D_params:
            p.data.add_(p_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != p_y.numel():
            raise RuntimeError('CG size mismatch')
        