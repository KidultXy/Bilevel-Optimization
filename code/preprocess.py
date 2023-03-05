"""
    MNIST数据相关的预处理函数
"""
import torch
from torchvision import datasets,transforms



def data_init(args):
    """
    MNIST数据集初始化
    """
    mu, std = 0.1307, 0.3081
    dataset = datasets.MNIST(root=args.data_path, train=True, download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((mu,), (std,))
                        ]))
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    # 训练集、测试集、方差降低法的小批量数据集
    train_set = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    vrsample_set = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.vrsample_size)
    test_set = torch.utils.data.DataLoader(datasets.MNIST(root=args.data_path, train=False,
                        download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((mu,), (std,))
                        ])), batch_size=args.test_size)
    return train_set, test_set, vrsample_set


def generate_noise(args, labels):
    """
    根据指定的noise_rate，生成标签的干扰信息
    """
    tmp = int(args.noise_rate * (labels.size()[0]))
    l_index = torch.randperm(labels.size()[0])[:tmp]
    labels[l_index] = (labels[l_index] + torch.randint(1, args.num_classes, (tmp,))) % args.num_classes
    return labels


def create_val_data(args, index, images, labels, device=torch.device('cpu')):
    """
    生成验证集数据
    """
    index = -index
    data = [[], []]
    for i in range(3):
        x, y = images[index[i]], labels[index[i]]
        x = torch.reshape(x, (x.size()[0],-1)).to(device)
        if i == 0:
            y = y.to(device)
        else:
            y = generate_noise(args, y).to(device)
        data[0].append(x)
        data[1].append(y)
    return data


def create_vrsample_data(args, images, labels, device=torch.device('cpu')):
    """
    生成方差降低法的小批量数据集
    """
    data = []
    for _ in range(args.iterations + 1):
        index = torch.randperm(args.train_size//args.vrsample_size)
        data.append(create_val_data(args, -index, images, labels, device))
    return data

