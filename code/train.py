"""
    训练过程所需函数
"""

import torch
from loss import *

# 内层函数输出
def f_output(data, i_params):
    output = torch.matmul(data, torch.t(i_params[:, 0:sep])) + i_params[:, sep]
    return output

# 输出正则化
def f_regularize(i_params, h_params, loss):
    ret = torch.mean(torch.mul(loss, torch.sigmoid(h_params))) + 0.001*torch.pow(torch.norm(i_params),2)
    return ret
