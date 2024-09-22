# coding=utf-8
import numpy as np
import struct
import os
import time

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01):  # 参数初始化
        # 权重初始化为正态分布
        # 在神经网络中，通常会选择一个较小的标准差，如0.01，以确保权重初始化不会太大。
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        # 偏置初始化为0
        self.bias = np.zeros([1, self.num_output])
    def forward(self, input):  # 前向传播计算
        start_time = time.time()
        self.input = input
        # TODO：全连接层的前向传播，计算输出结果
        self.output = np.dot(input, self.weight) + self.bias
        return self.output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        self.d_weight = np.dot(self.input.T, top_diff)  # 权重梯度
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)   # 偏置梯度
        bottom_diff = np.dot(top_diff, self.weight.T)  # 下一层的梯度
        return bottom_diff
    def update_param(self, lr):  # 参数更新
        # TODO：对全连接层参数利用参数进行更新
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):  # 参数保存
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input
        # TODO：ReLU层的前向传播，计算输出结果
        output = np.maximum(0, input)
        return output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：ReLU层的反向传播，计算本层损失
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff

class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')
    def forward(self, input):  # 前向传播的计算
        # TODO：softmax 损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob
    def get_loss(self, label):   # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算
        # TODO：softmax 损失层的反向传播，计算本层损失
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

# # 自写 Dropout 层
# class DropoutLayer(object):
#     def __init__(self, keep_prob):
#         self.keep_prob = keep_prob
#         self.mask = None

#     def forward(self, input):
#         # 创建一个与输入形状相同的随机数组，然后根据 keep_prob 决定是否保留该神经元
#         self.mask = np.random.rand(*input.shape) > (1 - self.keep_prob)
#         output = input * self.mask
#         return output

#     def backward(self, top_diff):
#         # 在反向传播时，确保 mask 与 top_diff 形状相同，并且正确地应用 mask
#         return top_diff * self.mask