"""
    fy、gy的梯度计算
"""

import torch
from torch.autograd import grad

def grad_fy(labels, i_params, output):
    loss = torch.nn.functional.cross_entropy(output, labels)
    grad = torch.autograd.grad(loss, i_params)[0]
    return grad

def grad_gy(labels, i_params, h_params, output, f_regularize):
    loss = torch.nn.functional.cross_entropy(output, labels, reduction='none')
    rloss = f_regularize(i_params, h_params, loss)
    grad = torch.autograd.grad(rloss, i_params, create_graph=True)[0]
    return grad
