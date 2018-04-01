# tensorflow教学示例

> 用最白话的语言，讲解机器学习、神经网络与深度学习
> 示例基于tensorflow1.4实现

## mnist
- [tensorflow入门-mnist手写数字识别(一，网络搭建)](https://geektutu.com/post/tensorflow-mnist-simplest.html)
    > [v1](mnist/v1)
    > 1. 这篇博客介绍了使用tensorflow搭建最简单的神经网络。
    > 2. 包括输入输出、独热编码与损失函数，以及正确率的验证。
- [tensorflow入门-mnist手写数字识别(二，模型保存加载)](https://geektutu.com/post/tensorflow-mnist-save-ckpt.html)
    > [v2](mnist/v2)
    > 1. 介绍了tensorflow中如何保存训练好的模型
    > 2. 介绍了如何从某一个模型为起点继续训练
    > 3. 介绍了模型如何加载使用，传入真实的图片如何识别
- [tensorflow入门-mnist手写数字识别(三，可视化训练)](https://geektutu.com/post/tensorflow-mnist-tensorboard-training.html)
    > [v3](mnist/v3)
    > 1. 介绍了tensorboard的简单用法，包括标量图、直方图以及网络结构图
- [tensorflow入门-mnist手写数字识别(四，h5py制作训练集)](https://geektutu.com/post/tensorflow-make-npy-hdf5-data-set.html)
    > [make_data_set](make_data_set)
    > 1. 介绍了如何使用numpy制作npy格式的数据集
    > 1. 介绍了如何使用h5py制作HDF5格式的数据集