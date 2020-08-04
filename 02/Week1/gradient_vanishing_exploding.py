import numpy as np
from matplotlib import pyplot as plt
from init_utils import *


# 定义权重的初始化函数，主要就是三种，
# W权重初始化为0
def initialize_params_zeros(layer_dims):
    params = {}
    L = len(layer_dims)
    for layer in range(1, L):
        params["W" + str(layer)] = np.zeros((layer_dims[layer], layer_dims[layer - 1]))
        params["b" + str(layer)] = np.zeros((layer_dims[layer], 1))

    return params


# W权重进行随机的初始化
def initialize_param_random(layer_dims):
    # np.random.seed(3)
    params = {}
    L = len(layer_dims)
    for layer in range(1, L):
        params["W" + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 10
        params["b" + str(layer)] = np.random.randn(layer_dims[layer], 1)

    return params


# He初始化
def initialize_param_he(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(
            2. / layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


# 定义模型
def model(train_X, train_Y, learning_rate, iterator_num, initialization):
    costs = []
    grad = {}
    m = train_X.shape[1]
    layer_dims = [train_X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        params = initialize_params_zeros(layer_dims)
    elif initialization == "random":
        params = initialize_param_random(layer_dims)
    elif initialization == "he":
        params = initialize_param_he(layer_dims)

    for i in range(iterator_num):
        AL, cache = forward_propagation(train_X, params)
        cost = compute_loss(AL, train_Y)
        grads = backward_propagation(train_X, train_Y, cache)
        params = update_parameters(params, grads, learning_rate)

        if i % 100 == 0:
            print("第", i, "次迭代，代价为:", cost)
        costs.append(cost)
    # 画代价曲线
    plt.plot(costs)
    plt.xlabel("iterator_num:")
    plt.ylabel("cost:")
    plt.title(str(initialization) + " initialization cost")
    plt.show()
    # 画拟合效果图
    plt.title(str(initialization) + " initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, np.squeeze(train_Y))

    return params


# 加载数据
train_X, train_Y, test_X, test_Y = load_dataset()
params = model(train_X, train_Y, learning_rate=0.01, iterator_num=10000, initialization="he")
print("训练集迭代次数:", 10000, "学习率:", 0.01)
Y_train_prediction = predict(train_X, train_Y, params)
print("\n测试集迭代次数:", 10000, "学习率:", 0.01)
Y_test_prediction = predict(test_X, test_Y, params)
