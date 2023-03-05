"""
    一个随机梯度下降（SGD）型优化器（stocBiO），是作者进行后续算法创新的基础
    其原理与实现由引文[20]所介绍
    
    Reference: 
    K. Ji, J. Yang, and Y. Liang. Bilevel optimization: Nonasymptotic analysis and faster algorithms.In International Conference on Machine Learning (ICML), 2021.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

from grad import *
from preprocess import *
from loss import *
from train import *


def stocBiO_sub(args, i_params, h_params, data, f_output, f_regularize):
    """
    随机梯度下降的过程
    依赖于Jacobian & Hessian Matrix
    -------
    
    """
    x, y = data
    # v0
    try:
        ot = f_output(x[0], i_params)
    except:
        ot = f_output(np.array(x), i_params)
    v0 = torch.unsqueeze(torch.reshape(grad_fy(y[0], i_params, ot), [-1]), 1).detach()
    # Hessian Matrix
    z = []
    ot = f_output(x[1], i_params)
    Grad_gy = grad_gy(y[1], i_params, h_params, ot, f_regularize) 
    Grad_G = torch.reshape(i_params, [-1]) - args.eta*torch.reshape(Grad_gy, [-1])
    for _ in range(args.hessianQ):
        Jacobian = torch.matmul(Grad_G, v0)
        v_new = torch.autograd.grad(Jacobian, i_params, retain_graph=True)[0]
        v0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
        z.append(v0)            
    v_Q = args.eta*v0 + torch.sum(torch.stack(z), dim=0)
    # G Gradient
    ot = f_output(x[2], i_params)
    Grad_gy = grad_gy(y[2], i_params, h_params, ot, f_regularize)
    Grad_gy = torch.reshape(Grad_gy, [-1])
    Grad_gyx = torch.autograd.grad(torch.matmul(Grad_gy, v_Q.detach()), h_params, retain_graph=True)[0]
    
    return -Grad_gyx


def stocBiO_train(args, device ,train_set, test_set):
    """
    基于stocBiO优化算法的训练过程
    
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
    o_index, t_index = 0, torch.randperm(args.batch_num) # 外层索引、训练集索引
    for epoch in range(args.epochs):
        i_grad = 0.0  # 一次epoch的inner梯度变化
        t_index = torch.randperm(args.batch_num) # 训练集索引
        for iter in range(args.iterations): # 迭代过程
            # 生成minibatch的数据
            m_index = t_index[iter % args.batch_num]
            images, labels = x[m_index], y[m_index]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels = generate_noise(args, labels).to(device)
            # 计算权重与一次迭代输出
            weight = gamma[m_index * args.batch_size: (m_index + 1) * args.batch_size]
            output = f_output(images, i_params) # inner-loop的一次输出
            i_update = grad_gy(labels, i_params, weight, output, f_regularize) # inner-loop中相关更新值
            i_params = i_params - args.inner_lr * i_update # inner-loop参数更新
            if iter == args.iterations - 1: # 最后一次迭代，输出梯度变化
                i_grad = torch.norm(i_update)
                print("Inner variation: {:.5f}".format(i_grad))

        v_index = torch.randperm(args.val_size//args.batch_size) # 验证集索引
        v_data = create_val_data(args, v_index, x, y, device) # 验证集数据
        h_params = gamma[o_index: o_index + args.batch_size] # outer的超参数
        o_update = stocBiO_sub(args, i_params, h_params, v_data, f_output, f_regularize) # outer-loop中相关更新值
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
    save_path = os.path.join(args.results_path, "stocBiO.npy")
    with open(save_path, 'wb') as f:
        np.save(f, history)


if __name__ == '__main__':
    
    from param_init import set_parameters
    args = set_parameters(optimizer="stocBiO")
    print("args:", args)

    train_set, test_set, _ = data_init(args)
    stocBiO_train(args, torch.device('cpu'), train_set, test_set)