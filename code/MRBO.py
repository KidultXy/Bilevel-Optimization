"""
    作者在论文中所论述的第一种优化算法：基于单环动量的递归二层优化器
    
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


def MRBO_train(args, device ,train_set, test_set):
    """
    基于MRBO优化算法的训练过程
    
    """
    # inner参数初始化
    i_params = torch.randn((args.num_classes, sep+1), requires_grad=True)
    i_params = nn.init.kaiming_normal_(i_params, mode='fan_out').to(device)
    # 超参数lamda初始化
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
        h_params = gamma[o_index: o_index + args.batch_size] # outer的超参数
        # 生成验证集数据与minibatch的数据
        v_index = -torch.randperm(args.train_size//args.batch_size)
        v_data = create_val_data(args, v_index, x, y, device)
        s_data, s_labels = v_data
        # 第k次迭代的相关参数初始化
        eta_k, alpha_k, beta_k, m = 1.0, 0.9, 0.9, 0.1

        if not epoch: # 初始
            grad_x = stocBiO_sub(args, i_params, h_params, v_data, f_output, f_regularize) # inner-loop的一次x梯度
            output = f_output(s_data[1], i_params) # inner-loop的一次输出
            grad_y = grad_gy(s_labels[1], i_params, s_data[1], h_params, output, f_regularize) # inner-loop的一次y梯度
            # inner-loop参数更新
            i_params, grad_y_old = i_params - args.inner_lr * eta_k * grad_y, grad_y
            #i_params = i_params 
        else: # 2...k，此时迭代要考虑先前态
            # k与k-1的inner-loop输出
            output = f_output(s_data[1], i_params)
            output_old = f_output(s_data[1], i_params_old)
            # k与k-1的update_y
            update_y = grad_gy(s_labels[1], i_params, s_data[1], h_params, output, f_regularize) 
            update_y_old = grad_gy(s_labels[1], i_params_old, s_data[1], h_params_old, output_old, f_regularize)
            grad_y = update_y + (1-beta_k) * (grad_y_old-update_y_old)
            # inner-loop参数更新
            i_params, grad_y_old = i_params, grad_y
            i_params = i_params - args.inner_lr * eta_k * grad_y
            # inner-loop中相关更新值
            update_x = stocBiO_sub(args, i_params, h_params, v_data, f_output, f_regularize)
            update_x_old = stocBiO_sub(args, i_params_old, h_params_old, v_data, f_output, f_regularize)
            grad_x = update_x + (1-alpha_k)*(grad_x_old-update_x_old)
        # 结束本次迭代前，用_old变量记录内层参数     
        i_params_old, h_params_old, grad_x_old, grad_y_old = i_params, h_params, grad_x, grad_y
        i_params, h_params = i_params - args.inner_lr * eta_k * grad_y, h_params - args.outer_lr * eta_k * grad_x
        o_update = torch.squeeze(o_update)
        i_grad = torch.norm(grad_y)
        # 结束本次迭代前，更新weight、eta、alpha、beta
        weight = h_params
        eta_k = eta_k*(((epoch+m)/(epoch+m+1))**(1/3))
        alpha_k, beta_k = alpha_k*(eta_k**2), beta_k*(eta_k**2)
        # 输出本次迭代的优化结果
        print("Inner variation: {:.5f}".format(i_grad))
        print("Outer variation: {:.5f}".format(torch.norm(grad_x)))

        with torch.no_grad():
            weight = weight - args.outer_lr * o_update # 更新权重
            gamma[o_index: o_index + args.batch_size] = weight # 将更新后权重赋值给超参数lamda_x
            o_index = (o_index + args.batch_size) % args.train_size
        # 训练集与测试集loss计算
        train_loss = cal_train_loss(train_loader, i_params, device, args.batch_num)
        test_loss = cal_test_loss(test_loader, i_params, device)
        time_cost = time.time() - start_time
        print('Epoch: {:d} Train loss: {:.5f} Test loss: {:.5f} Time: {:.5f}'.format(epoch+1, train_loss, test_loss, time_cost))

        history[epoch+1, :] = [train_loss, test_loss, time_cost, float(i_grad)]

    print(history)
    save_path = os.path.join(args.results_path, "MRBO.npy")
    with open(save_path, 'wb') as f:
        np.save(f, history)


if __name__ == '__main__':
    
    from param_init import set_parameters
    args = set_parameters("MRBO")
    print("args:", args)

    train_loader, test_loader, vrsample_loader = data_init(args)
    MRBO_train(args, torch.device('cpu'), train_loader, test_loader)