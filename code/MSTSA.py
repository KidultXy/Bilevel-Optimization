"""
    Basiline算法之一：MSTSA
    [22]提出了一种通过[3,36]引入的基于动量的递归技术来更新x的算法MSTSA
    Reference:P. Khanduri, S. Zeng, M. Hong, H.-T. Wai, Z. Wang, and Z. Yang. A momentum-assisted single-timescale stochastic approximation algorithm for bilevel optimization. arXiv preprint arXiv:2102.07367v1, 2021.
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn

from grad import *
from preprocess import *
from loss import *
from train import *
from stocBiO import stocBiO_sub

def MSTSA_sub(args, o_update_old, c_eta, v_data, i_params, i_params_new, h_params, h_params_old, f_output, f_regularize):
    """
    MSTSA算法更新outer梯度的过程
    
    """
    grad_new = stocBiO_sub(args, i_params_new, h_params, v_data, f_output, f_regularize)
    grad_pri = stocBiO_sub(args, i_params, h_params_old, v_data, f_output, f_regularize)
    o_update = c_eta * grad_new + (1 - c_eta) * (o_update_old + grad_new - grad_pri)
    return o_update


def MSTSA_train(args, device ,train_set, test_set):
    """
    基于MSTSA优化算法的训练过程
    
    """
    # inner参数初始化
    i_params = torch.randn((args.num_classes, sep+1), requires_grad=True)
    i_params = nn.init.kaiming_normal_(i_params, mode='fan_out').to(device)
    # 超参数gamma初始化
    gamma = torch.zeros((args.train_size), requires_grad=True, device=device).to(device)
    # 初始loss值
    train_loss, test_loss = cal_train_loss(train_set, i_params, device, args.batch_num), cal_test_loss(test_set, i_params, device)
    print('Initial: Train loss: {:.4f} Test loss: {:.4f}'.format(train_loss, test_loss))
    # 训练过程的记录
    history = np.zeros((args.epochs + 1, 4)) # 4列数据
    history[0, :] = [train_loss.cpu(), test_loss.cpu(), (0.0), (0.0)]
    # 数据列与标签列拆分
    x, y = [], []
    for img, label in train_set:
        x.append(img)
        y.append(label)
    # 计时器
    start_time = time.time() 
    o_index, o_update_old = 0, 0 # 外层索引、上次外层梯度
    for epoch in range(args.epochs):
        i_grad = 0.0  # 一次epoch的inner梯度变化
        h_params = gamma[o_index: o_index + args.batch_size] # outer的超参数
        if epoch == 0:
            o_update_old = 0
            h_params_old = h_params
        else:
            h_params_old = gamma[o_index - args.batch_size: o_index]
        c_eta = 0.5 # 参数默认值
        # 学习率递减
        args.outer_lr = 0.1/(math.sqrt(epoch+1))
        beta_t = args.inner_lr/(math.sqrt(epoch+1))
        # 生成minibatch的数据
        images, labels = x[epoch % args.batch_num], y[epoch % args.batch_num]
        images = torch.reshape(images, (images.size()[0],-1)).to(device)
        labels = generate_noise(args, labels).to(device)    
        ot = f_output(images, i_params) # inner-loop的一次输出
        i_update = grad_gy(labels, i_params, h_params, ot, f_regularize) # inner-loop中相关更新值
        i_params_new = i_params - beta_t * i_update
        
        v_index = torch.randperm(args.val_size//args.batch_size) # 验证集索引
        v_data = create_val_data(args, v_index, x, y, device) # 验证集数据
        o_update = MSTSA_sub(args, o_update_old, c_eta, v_data, i_params, i_params_new, h_params, h_params_old, f_output, f_regularize)
        i_params = i_params_new

        # 计算权重与一次迭代输出
        weight = h_params
        o_update = torch.squeeze(o_update) # outer-loop中相关更新值
        print("Outer variation: {:.5f}".format(torch.norm(o_update)))
        
        with torch.no_grad():
            weight = weight - args.outer_lr * torch.squeeze(o_update) # 更新权重
            gamma[o_index: o_index + args.batch_size] = weight # 将更新后权重赋值给超参数gamma
            o_index = (o_index + args.batch_size) % args.train_size
        # 训练集与测试集loss计算
        train_loss = cal_train_loss(train_set, i_params, device, args.batch_num)
        test_loss = cal_test_loss(test_set, i_params, device)
        time_cost = time.time() - start_time # 记录训练用时
        print('Epoch: {:d} Train loss: {:.5f} Test loss: {:.5f} Time cost: {:.5f}'.format(epoch + 1, train_loss, test_loss, time_cost))

        history[epoch+1, :] = [train_loss, test_loss, time_cost, float(i_grad)]

    print("history:", history)
    save_path = os.path.join(args.results_path, "MSTSA.npy")
    with open(save_path, 'wb') as f:
        np.save(f, history)


if __name__ == '__main__':
    
    from param_init import set_parameters
    args = set_parameters(optimizer="MSTSA")
    print("args:", args)

    train_set, test_set, _ = data_init(args)
    MSTSA_train(args, torch.device('cpu'), train_set, test_set)