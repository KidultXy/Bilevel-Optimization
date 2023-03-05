"""
    作者在论文中所论述的第二种优化算法：基于方差减少技术的随机双层优化算法
    
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

from stocBiO import *
from grad import *
from preprocess import *
from loss import *
from train import *


def VRBO_iter(args, v_data, i_params, h_params, h_params_old, grad_x, grad_y):
    """
    VRBO算法的一轮迭代过程

    """
    x, y = v_data[0]
    ot = f_output(x[1], i_params) # inner-loop层输出
    # 初始xy的更新值
    dy = grad_gy(y[1], i_params, h_params, ot, f_regularize) 
    dy_old = grad_gy(y[1], i_params, h_params_old, ot, f_regularize)
    dx = stocBiO_sub(args, i_params, h_params, v_data[0], f_output, f_regularize)
    dx_old = stocBiO_sub(args, i_params, h_params_old, v_data[0], f_output, f_regularize)
    # 构造用于更新x和y的递归方差减少梯度估计器
    v_t, u_t = grad_x + dx - dx_old, grad_y + dy - dy_old
    # 更新并记录参数信息
    i_params_new = i_params - args.inner_lr * u_t
    for t in range(args.iterations):
        # 生成minibatch的数据
        x, y = v_data[t + 1]
        ot = f_output(x[1], i_params_new) # inner-loop层输出
        # 迭代中x、y的更新值，对应原文算法描述中的xk xk-1
        dy = grad_gy(y[1], i_params_new, h_params, ot, f_regularize)
        ot = f_output(x[1], i_params)
        dy_old = grad_gy(y[1], i_params, h_params, ot, f_regularize)
        dx = stocBiO_sub(args, i_params_new, h_params, v_data[t+1], f_output, f_regularize)
        dx_old = stocBiO_sub(args, i_params, h_params, v_data[t+1], f_output, f_regularize)
        # 构造用于更新x和y的递归方差减少梯度估计器
        v_t = v_t + dx - dx_old 
        u_t = u_t + dy - dy_old
        i_params = i_params_new
        i_params_new = i_params - args.inner_lr * u_t # 更新并记录参数信息
    # 返回最终的参数信息与梯度估计器
    return i_params_new, v_t, u_t


def VRBO_train(args, device ,train_set, test_set, vrsample_set):
    """
    基于VRBO优化算法的训练过程
    
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
    vx, vy = [], []
    for img, label in vrsample_set:
        vx.append(img)
        vy.append(label)
    # 计时器
    start_time = time.time() 
    o_index, t_index = 0, torch.randperm(args.batch_num) # 外层索引、训练集索引

    for epoch in range(args.epochs):
        i_grad = 0.0  # 一次epoch的inner梯度变化
        if epoch % args.vr_epoch == 0:
            h_params = gamma[o_index: o_index + args.batch_size] # outer的超参数
            v_index = torch.randperm(args.val_size//args.batch_size) # 验证集索引
            v_data = create_val_data(args, v_index, x, y, device) # 验证集数据
            grad_x = stocBiO_sub(args, i_params, h_params, v_data, f_output, f_regularize)
            ot = f_output(v_data[0][1], i_params) # inner-loop的一次输出
            grad_y = grad_gy(v_data[1][1], i_params, h_params, ot, f_regularize)

            h_params = gamma[o_index + args.batch_size: o_index + args.batch_size + args.vrsample_size] # gamma的区间更新outer的超参数
            h_params_old = gamma[o_index - args.vrsample_size + args.batch_size: o_index + args.batch_size] # 记录原有的outer参数
        else:
            h_params = gamma[o_index: o_index + args.vrsample_size] # gamma的区间更新outer的超参数
            h_params_old = gamma[o_index - args.vrsample_size: o_index] # 记录原有的outer参数

        v_data = create_vrsample_data(args, vx, vy, device) # 验证集数据
        i_params_new, grad_x, grad_y = VRBO_iter(args, v_data, i_params, h_params, h_params_old, grad_x[0:args.vrsample_size], grad_y)
        i_params = i_params_new # inner-loop参数更新
        i_grad = torch.norm(grad_y)
        weight = h_params
        o_update = torch.squeeze(grad_x)

        # 输出本次epoch的Inner&Outer的提升
        print("Inner variation: {:.5f}".format(i_grad))
        print("Outer variation: {:.5f}".format(torch.norm(grad_x)))
        with torch.no_grad():
            weight = weight - args.outer_lr * o_update # 更新权重
            gamma[o_index: o_index + args.vrsample_size] = weight # 将更新后权重赋值给超参数gamma
            o_index = (o_index + args.vrsample_size) % args.train_size
        # 训练集与测试集loss计算
        train_loss = cal_train_loss(train_set, i_params, device, args.batch_num)
        test_loss = cal_test_loss(test_set, i_params, device)
        time_cost = time.time() - start_time
        print('Epoch: {:d} Train loss: {:.5f} Test loss: {:.5f} Time: {:.5f}'.format(epoch+1, train_loss, test_loss, time_cost))

        history[epoch+1, :] = [train_loss, test_loss, time_cost, float(i_grad)]

    print(history)
    save_path = os.path.join(args.results_path, "VRBO.npy")
    with open(save_path, 'wb') as f:
        np.save(f, history)


if __name__ == '__main__':
    
    from param_init import set_parameters
    args = set_parameters("VRBO")
    print("args:", args)

    train_loader, test_loader, vrsample_loader = data_init(args)
    VRBO_train(args, torch.device('cpu'), train_loader, test_loader, vrsample_loader)