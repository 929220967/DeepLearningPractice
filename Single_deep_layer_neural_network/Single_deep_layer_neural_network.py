# -*- coding: utf-8 -*-
"""
    Created on Thu Jul 30 20:29:27 2020
    @Author Mr.Lu
"""
import numpy as np
from planar_utils import *
from matplotlib import pyplot as plt
from testCases import *

"""
    在写代码之前，我们需要明白我们需要做什么：
    1、构建一个具有单隐藏层的二分类网络
    2、使用非线性激活函数激活函数，例如使用tanh
    3、计算交叉损失熵，也就是计算代价函数
    4、实现向前传播或者向后传播
"""
"""
    代码思路：
    1、加载数据集
    2、定义神经网络结构
    3、初始化参数
    4、定义向前传播函数
    5、定义计算代价的函数    
    6、定义反向传播的函数
    7、更新参数
    8、定义模型函数
    9、定义预测函数
"""
np.random.seed(1)


# 2、定义神经网络的结构
def layers_size(X, Y, n_h):
    n_x = X.shape[0]  # 定义的输入特征的数量 为2
    # n_h = 4  # 定义隐藏层的节点数量  4个
    n_y = Y.shape[0]  # 定义输出层的节点个数 为1
    layers_size = {
        "n_x": n_x,
        "n_h": n_h,
        "n_y": n_y
    }
    return layers_size


# 3、初始化参数
def initalize_parameters(layers_sieze):
    np.random.seed(2)
    n_x = layers_sieze["n_x"]  # 输入的特征数量
    n_h = layers_sieze["n_h"]  # 隐藏层的节点个数
    n_y = layers_sieze["n_y"]  # 输出层的节点个数
    W1 = np.random.randn(n_h, n_x) * 0.01  # 初始化隐藏层的权重参数
    b1 = np.zeros((n_h, 1))  # 初始化隐藏层的偏移量
    W2 = np.random.randn(n_y, n_h) * 0.01  # 初始化输出层的权重参数
    b2 = np.zeros((n_y, 1))  # 初始化输出层的偏移量

    parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }

    return parameters


# 4、定义向前传播函数
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)  # 隐藏层使用tanh作为激活函数
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # 输出层使用sigmoid函数作为激活函数
    cache = {
        "Z1": Z1,
        "Z2": Z2,
        "A1": A1,
        "A2": A2
    }
    return A2, cache


# 5、定义计算代价的函数
def compute_cost(A2, Y, parameters):
    m = A2.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))  # 也就是误差值

    return cost


# 6、定义反向传播的函数
def backward_propagation(cache, X, Y, parameters):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A2 = cache["A2"]
    A1 = cache["A1"]
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "dW2": dW2,
        "db1": db1,
        "db2": db2
    }
    return grads


# 7、更新参数
def update_parameters(parameters, learning_rate, grads):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }

    return parameters


# 8、定义模型函数
def nn_model(X, Y, iterator_nums, learning_rate, n_h):
    np.random.seed(3)
    layers_nums = layers_size(X, Y, n_h)
    parameters = initalize_parameters(layers_nums)

    costs = []  # 画代价曲线
    for i in range(iterator_nums):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(cache, X, Y, parameters)
        parameters = update_parameters(parameters, learning_rate, grads)
        # print("第", i, "次迭代，损失为:", cost)
        if i % 100 == 0:
            costs.append(cost)

    return costs, parameters


# 9、定义预测函数
def prediction(X, parameters):
    A2, cache = forward_propagation(X, parameters)
    prediction = np.round(A2)
    return prediction


# 1、加载数据集
X, Y = load_planar_dataset()  # X的维数为2*400，Y的维数为1*400
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral) #绘制散点图
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]  # 隐藏层数量

costALL = {}  # 保存画代价曲线的数据

for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    costs, parameters = nn_model(X, Y, iterator_nums=10000, learning_rate=0.5, n_h=n_h)
    costALL[str(i)] = costs  # 保存画代价曲线的数据
    plot_decision_boundary(lambda x: prediction(x.T, parameters), X, Y)
    predictions = prediction(X, parameters)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    # print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
    print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
plt.show()

# 画学习率的指示线
for i in range(len(hidden_layer_sizes)):
    plt.plot(np.squeeze(costALL[str(i)]), label=str(hidden_layer_sizes[i]))
plt.title(" learning_rate:0.05")
plt.xlabel("iterators_num")
plt.ylabel("cost")
legend = plt.legend(loc='upper right', shadow=True)  # 设置解释所在位置
frame = legend.get_frame()  # 在右上角覆盖一个小窗口，显示学习率的解释
frame.set_facecolor('0.90')  # 设置右上角解释窗口的背景颜色的深度
plt.show()
