# -*- coding: utf-8 -*-
"""
    Created on Wed Jul 29 21:04:27 2020
    @Author Mr.Lu
"""
import numpy as np
from lr_utils import load_dataset
from matplotlib import pyplot as plt

"""
    1、加载数据
    2、将输入的训练集特征变量数据降维并转置，变成12288*209维
    3、将测试集的输入特征变量数据降维并转置，变成12288*50维
    4、输入的特征数据归一化处理，因为图片的像素颜色是由r、g、b三原色构成，只能构成256种，
       因此在数据进行归一化的时候，需要将数据除以255
    5、初始化参数 
    6、定义激活函数
    7、定义正向传播和反向传播的函数
    8、定义梯度下降的函数，在函数内部进行权值W和偏移量b的修正
    9、定义预测函数
    10、执行函数
"""


# 5、初始化参数
def initalize_parameters(layers_dims):  # layers_dims:隐藏层的节点数
    W = np.zeros((layers_dims, 1))
    b = 0  # 偏移量

    return W, b


# 6、定义激活函数
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


# 7、定义正向传播和反向传播的函数
def forward_backward_propagation(X, Y, W, b, ):
    m = X.shape[1]
    A = sigmoid(np.dot(W.T, X) + b)

    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # 反向传播
    dW = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    grads = {
        "dW": dW,
        "db": db,
        "cost": cost
    }
    return grads


# 8、定义梯度下降的函数，在函数内部进行权值W和偏移量b的修正
def optamize(X, Y, W, b, iterator_num, learning_Rate):
    costs = []

    for i in range(iterator_num):
        grads = forward_backward_propagation(X, Y, W, b)
        dW = grads["dW"]
        db = grads["db"]
        cost = grads["cost"]

        W = W - learning_Rate * dW
        b = b - learning_Rate * db
        # print("第", i, "次迭代，代价值为:", cost)
        if i % 100 == 0:
            costs.append(cost)

    parameters = {
        "W": W,
        "b": b,
        "costs": costs
    }
    return parameters


# 9、定义预测函数
def predict(X, W, b):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    W = W.reshape(X.shape[0], 1)
    A = np.dot(W.T, X) + b
    A = sigmoid(A)
    for i in range(m):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction


# 10、执行函数
def model(train_x, train_y, test_x, test_y, iterator_nums, learning_rate):
    W, b = initalize_parameters(train_x.shape[0])  # 12288个
    parameters = optamize(train_x, train_y, W, b, iterator_nums, learning_rate)
    W = parameters["W"]
    b = parameters["b"]
    Y_prediction_test = predict(test_x, W, b)
    Y_prediction_train = predict(train_x, W, b)
    print("训练集准确性: ", format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100), "%")
    print("测试集准确性: ", format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100), "%")
    d = {
        "W": W,
        "b": b,
        "costs": parameters["costs"],
        "train_prediction": Y_prediction_train,
        "test_prediction": Y_prediction_test,
        "learning_rate": learning_rate
    }

    return d


train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()  # 1、加载数据
# 2、将输入的训练集特征变量数据降维并转置，变成12288*209维
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # 12288*209
# 3、将测试集的输入特征变量数据降维并转置，变成12288*50维
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T  # 12288*50
# 4、输入的特征数据归一化处理，因为图片的像素颜色是由r、g、b三原色构成，只能构成256种，
#   因此在数据进行归一化的时候，需要将数据除以255
train_set_x = train_set_x_flatten / 255  # 12288*209
test_set_x = test_set_x_flatten / 255  # 12288*50

learning_rate = [0.0001, 0.001, 0.01, 0.005, 0.007, 0.015]  # 定义一个学习率的数组
models = {}  # 定义一个列表，用来保存不同学习率的返回数据
# 按照不同的学习率执行
for i in range(len(learning_rate)):
    print("learning_rate:", learning_rate[i])
    models[str(i)] = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, iterator_nums=1500,
                           learning_rate=learning_rate[i])
    print("=========================================\n")
# 画学习率的指示线
for i in range(len(learning_rate)):
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')  # 纵坐标名称
plt.xlabel('iteration')  # 横坐标名称
plt.title("Deep Learning")  # 设置图表标题
legend = plt.legend(loc='upper right', shadow=True)  # 设置解释所在位置
frame = legend.get_frame()  # 在右上角覆盖一个小窗口，显示学习率的解释
frame.set_facecolor('0.90')  # 设置右上角解释窗口的背景颜色的深度
plt.show()  # 显示图表
