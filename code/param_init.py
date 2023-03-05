"""
    参数定义
    默认适用于minist数据的十分类问题
"""

import argparse
import os

def set_parameters(optimizer):
    parser = argparse.ArgumentParser()
    # 优化算法的选择
    parser.add_argument('--optimizer', type=str, default=optimizer)
    # 十分类
    parser.add_argument('--num_classes', type=int, default=10)
    # size
    parser.add_argument('--batch_size', type=int, default=480)
    parser.add_argument('--test_size', type=int, default=480)
    parser.add_argument('--train_size', type=int, default=20000)
    parser.add_argument('--val_size', type=int, default=5000)
    parser.add_argument('--vrsample_size', type=int, default=32)
    parser.add_argument('--batch_num', type=int, default=0)
    # 迭代
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--vr_epoch', type=int, default=3)
    # 参数设定
    parser.add_argument('--outer_lr', type=float, default=0.1)
    parser.add_argument('--inner_lr', type=float, default=0.1)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--noise_rate', type=float, default=0.1)
    parser.add_argument('--hessianQ', type=int, default=3)
    # 路径
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--results_path', type=str, default='./results')

    ret = parser.parse_args()
    ret.batch_num = ret.train_size//ret.batch_size
    ret.results_path = os.path.join(ret.results_path, ret.optimizer)
    if not os.path.isdir(ret.results_path):
        os.makedirs(ret.results_path)
    return ret