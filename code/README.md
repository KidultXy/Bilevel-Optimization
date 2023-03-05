## 论文复现：Provably Faster Algorithms for Bilevel Optimization

- 环境说明：

  python   3.9.7

  torch   1.8.0+cu111

  torchvision   0.10.1

  numpy    1.22.4

- 目录：

  data：数据集存放，默认MMIST

  model：保存的部分训练模型

  output：不同优化器的在训练过程中收敛速度、loss变化等

  results：模型在断点处参数的记录

- 文件：

  param_init.py：参数设定及初始化（借鉴了作者的实现思路）

  preprocess.py：针对MNIST的数据预处理函数

  grad.py：inner-loop中fy、gy梯度的相关运算

  loss.py：损失函数的定义、针对训练集与测试集的损失计算

  train.py：模型训练的若干通用函数

- 算法

  stocBiO.py：一个随机梯度下降（SGD）型优化器（stocBiO），是作者进行后续算法创新的基础

  MRBO.py：作者在论文中所论述的第一种优化算法：基于单环动量的递归二层优化器

  VRBO.py：作者在论文中所论述的第二种优化算法：基于方差减少技术的随机双层优化算法

  MSTSA.py：Basiline算法之一：MSTSA

  Basiline.py：Basiline算法之一：HOAG