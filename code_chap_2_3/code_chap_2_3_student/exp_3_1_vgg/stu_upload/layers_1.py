import numpy as np
import struct
import os
import time

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

class FullyConnectedLayer(object):  # 全连接层初始化
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01):   # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')
    def forward(self, input):   # 前向传播计算
        start_time = time.time()
        self.input = input
        # TODO：全连接层的前向传播，计算输出结果
        # self.input.dot(self.weight)计算了输入与权重的点积（即矩阵乘法），然后加上偏置向量，得到的结果赋值给self.output，即当前层的输出
        self.output = self.input.dot(self.weight) + self.bias
        return self.output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        # 根据链式法则，权重梯度等于输入与上一层梯度的乘积
        self.d_weight = np.matmul(self.input.T, top_diff)
        # 创建一个全为1的行向量（形状为[1, top_diff.shape[0]]），然后与top_diff做矩阵乘法
        # 结果是偏置梯度的总和，这个值对于每个输入都是相同的，因此我们可以简单地将top_diff的总和作为偏置的梯度
        self.d_bias = np.matmul(np.ones([1, top_diff.shape[0]]), top_diff)
        # 损失函数对这一层输入的梯度
        bottom_diff = np.matmul(top_diff, self.weight.T)
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):  # 参数更新
        # TODO：对全连接层参数利用参数进行更新
        # 根据梯度下降算法更新权重参数
        # 计算梯度的负值（因为我们要沿着损失函数减小的方向更新参数）乘以学习率，然后从当前权重中减去这个值，以得到新的权重值
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self): # 参数保存
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')
    def forward(self, input):  # 前向传播的计算
        self.input = input
        # TODO：ReLU层的前向传播，计算输出结果
        # ReLU函数是一个非线性激活函数，它的定义是f(x) = max(0, x)
        output = np.maximum(0, self.input)
        return output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：ReLU层的反向传播，计算本层损失
        # 当输入为正时，梯度保持不变；当输入为负或零时，梯度变为零
        bottom_diff = top_diff * (self.input >= 0.)
        return bottom_diff

class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')
    def forward(self, input):  # 前向传播的计算
        # TODO：softmax 损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        # 将上一步得到的指数结果input_exp按行进行归一化
        # 使得每一行变成一个概率分布，表示了对应样本属于各个类别的概率
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob
    def get_loss(self, label): # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算
        # TODO：softmax 损失层的反向传播，计算本层损失
        # 计算预测概率和真实标签（one-hot编码）之间的差异
        # 并将差异平均到每个样本上
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

