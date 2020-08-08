# _*_ coding: utf-8 _*_
"""
    Created on Fri Aug 7 2020 10:54:27
    @Author Mr.Lu

"""

import numpy as np
from matplotlib import pyplot as plt
from opt_utils import *
# from testCases import *
import math

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
"""
    在进行编程之前，我们应该首先明白要做什么。在这个程序中，我们需要做的是数据集的分割工作和优化梯度下降算法。
    因此需要做以下的工作：
    1、分割数据集
    2、优化梯度下降算法：
       2.1、不适应任何优化算法
       2.2、使用mini-batch梯度下降法
       2.3、使用具有动量的梯度下降法
       2.4、使用Adam算法    
"""
# 1、加载数据
train_X, train_Y = load_dataset()
layer_dims = [train_X.shape[0], 5, 2, 1]
optimizeres = ["gd", "momentum", "adam"]


# 2、没有做任何优化的梯度下降进行参数更新
def update_parameters_with_gd(parameters, grads, learning_rate):
    m = len(parameters) // 2
    for i in range(m):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]

    return parameters


# 3、分割数据集，形成多个不同的mini-batch
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # 第一步，打乱顺序
    permutation = list(np.random.permutation(m))  # 返回一个长度为m的随机数组，里面的数是0~m-1
    shuffled_X = X[:, permutation]  # 将每一列的数据按打乱的数据来重新排列
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # 第二步，分割
    num_complete_mini_batches = math.floor(m / mini_batch_size)  # 返回一个给定的数字的最大整数
    for k in range(0, num_complete_mini_batches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 如果不能整体分割，就把剩下的作为一个整体打包成一个mini_batch块
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_mini_batches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_mini_batches:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# 4、包含动量的梯度下降，即梯度的指数加权平均值
def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for i in range(L):
        v["dW" + str(i + 1)] = np.zeros_like(parameters["W" + str(i + 1)])
        v["db" + str(i + 1)] = np.zeros_like(parameters["b" + str(i + 1)])

    return v


# 5、影响梯度的方向
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    m = len(parameters) // 2
    for i in range(m):
        v["dW" + str(i + 1)] = beta * v["dW" + str(i + 1)] + (1 - beta) * grads["dW" + str(i + 1)]
        v["db" + str(i + 1)] = beta * v["db" + str(i + 1)] + (1 - beta) * grads["db" + str(i + 1)]

        # 更新参数
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * v["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * v["db" + str(i + 1)]

    return parameters, v


# 6、Adam算法
def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for i in range(L):
        v["dW" + str(i + 1)] = np.zeros_like(parameters["W" + str(i + 1)])
        v["db" + str(i + 1)] = np.zeros_like(parameters["b" + str(i + 1)])

        s["dW" + str(i + 1)] = np.zeros_like(parameters["W" + str(i + 1)])
        s["db" + str(i + 1)] = np.zeros_like(parameters["b" + str(i + 1)])

    return (v, s)


# 7、更新参数
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for i in range(L):
        v["dW" + str(i + 1)] = beta1 * v["dW" + str(i + 1)] + (1 - beta1) * grads["dW" + str(i + 1)]
        v["db" + str(i + 1)] = beta1 * v["db" + str(i + 1)] + (1 - beta1) * grads["db" + str(i + 1)]

        # 修正偏差
        v_corrected["dW" + str(i + 1)] = v["dW" + str(i + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(i + 1)] = v["db" + str(i + 1)] / (1 - np.power(beta1, t))

        s["dW" + str(i + 1)] = beta2 * s["dW" + str(i + 1)] + (1 - beta2) * np.square(grads["dW" + str(i + 1)])
        s["db" + str(i + 1)] = beta2 * s["db" + str(i + 1)] + (1 - beta2) * np.square(grads["db" + str(i + 1)])

        s_corrected["dW" + str(i + 1)] = s["dW" + str(i + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(i + 1)] = s["db" + str(i + 1)] / (1 - np.power(beta2, t))

        # 更新权重
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * (v_corrected["dW" + str(i + 1)] / (np.sqrt(s_corrected["dW" + str(i + 1)]) + epsilon))
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * (v_corrected["db" + str(i + 1)] / np.sqrt(s_corrected["db" + str(i + 1)] + epsilon))

    return (parameters, v, s)


# 8、定义模型
def model(X, Y, layer_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, iterators_num=10000):
    L = len(layer_dims)
    costs = []
    t = 0
    seed = 10

    parameters = initialize_parameters(layer_dims)

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    else:
        print("error!")
        exit(1)

    # 开始学习
    for i in range(iterators_num):
        seed = seed + 1

        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            minibatch_X, minibatch_Y = minibatch
            A3, cache = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(A3, minibatch_Y)

            grads = backward_propagation(minibatch_X, minibatch_Y, cache)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,
                                                               epsilon)

        if i % 100 == 0:
            costs.append(cost)
            print("第", i, "次迭代，代价为:", cost)

    plt.plot(costs)
    plt.title("Learning rate = " + str(learning_rate) + " Model with " + optimizer + " optimization")
    plt.xlabel("iterators_num")
    plt.ylabel("cost")
    plt.show()

    return parameters


# 9、将三种情况都分别执行一遍
for optimizer in optimizeres:
    parameter = model(train_X, train_Y, layer_dims, optimizer)
    predictions = predict(train_X, train_Y, parameter)
    plt.title("Model with " + optimizer + " optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameter, x.T), train_X, np.squeeze(train_Y))
