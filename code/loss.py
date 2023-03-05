"""
    loss计算的相关函数
"""

import torch

sep = 784

# 损失函数
def f_loss(labels, i_params, data):
    output = torch.matmul(data, torch.t(i_params[:, 0:sep])) + i_params[:, sep]
    return torch.nn.functional.cross_entropy(output, labels)

# 计算训练集loss
def cal_train_loss(data_loader, i_params, device, batch_num):
    total, count = 0.0, 0
    for index, (images, labels) in enumerate(data_loader):
        if index >= batch_num:
            break
        else:
            images, labels = torch.reshape(images, (images.size()[0],-1)).to(device), labels.to(device)
            loss = f_loss(labels, i_params, images)
            total += loss 
            count += 1
    total = total/count
    return total.detach()

# 计算测试集loss
def cal_test_loss(data_loader, i_params, device):
    total, count = 0.0, 0
    for images, labels in data_loader:
        images, labels = torch.reshape(images, (images.size()[0],-1)).to(device), labels.to(device)
        loss = f_loss(labels, i_params, images)
        total += loss
        count += 1
    total = total / count
    return total.detach()


