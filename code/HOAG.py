"""
    Basiline算法之一：HOAG
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


def HOAG_train(args, device ,train_set, test_set):
    """
    基于HOAG优化算法的训练过程
    
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
        # 生成minibatch的数据
        images, labels = x[-1], y[-1]
        images = torch.reshape(images, (images.size()[0],-1)).to(device)
        images_temp, labels_temp = images[0:args.val_size,:], labels[0:args.val_size]
        images, labels = torch.cat([images_temp]*(args.train_size // args.val_size)), torch.cat([labels_temp]*(args.train_size // args.val_size))
        labels = labels.to(device)
        # Fy梯度的计算
        ot = f_output(images, i_params)
        grad_Fy = grad_fy(labels, i_params, ot)
        v0 = torch.unsqueeze(torch.reshape(grad_Fy, [-1]), 1).detach()
        # Hessian Matrix 的计算
        z = []
        v_Q = args.eta * v0
        labels = generate_noise(args, labels).to(device)
        h_params = gamma[o_index: o_index + args.batch_size]
        ot = f_output(images, i_params)
        grad_Gy = grad_gy(labels, i_params, h_params, ot, f_regularize)
        grad_Gy = torch.reshape(grad_Gy, [-1])
        grad_G = torch.reshape(i_params, [-1]) - args.eta * grad_Gy
        for _ in range(args.hessian_q):
            Jacobian = torch.matmul(grad_G, v0)
            vnew = torch.autograd.grad(Jacobian, i_params, retain_graph=True)[0]
            v0 = torch.unsqueeze(torch.reshape(vnew, [-1]), 1).detach()
            z.append(v0)
        # 更新vQ
        vQ += torch.sum(torch.stack(z), dim=0)
        o_update = - torch.autograd.grad(torch.matmul(grad_Gy, vQ.detach()), h_params)[0] # outer-loop中相关更新值
        o_update = torch.squeeze(o_update) # outer-loop中相关更新值
        print("Outer variation: {:.5f}".format(torch.norm(o_update)))
        
        # 计算权重与一次迭代输出
        weight = h_params
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
    save_path = os.path.join(args.results_path, "HOAG.npy")
    with open(save_path, 'wb') as f:
        np.save(f, history)


if __name__ == '__main__':
    
    from param_init import set_parameters
    args = set_parameters(optimizer="HOAG")
    print("args:", args)

    train_set, test_set, _ = data_init(args)
    HOAG_train(args, torch.device('cpu'), train_set, test_set)