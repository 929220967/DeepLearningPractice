import numpy as np
from scipy import ndimage

from lr_utils import load_dataset
from dnn_utils import *
from testCases import *
from matplotlib import pyplot as plt
import scipy
from matplotlib.pyplot import imread
from PIL import Image
from scipy import ndimage
"""
    在进行编程之前，我们应该首先明白要做什么：
    1、构建一个多层的神经网络
    2、初始化神经网络参数
    3、使用多种激活函数
    4、计算交叉损失熵，即代价函数
    5、定义向前传播和反向传播的函数
    6、更新参数
    在隐藏层使用的激活函数是relu函数，在输出层使用的是sigmoid函数，
    因此在进行向前传播和反向传播的时候要分类进行
"""
"""
    代码思路：
    1、加载数据,并且对数据进行归一化处理
    2、初始化神经网络的参数
    3、定义神经网络结构
    4、定义计算函数
    5、定义向前传播的分类函数，以此来判断是使用relu激活函数还是sigmoid函数
    6、定义多层的整个神经网络的向前传播的计算函数
    7、定义代价函数
    8、定义反向传播的计算函数
    9、定义反向传播的分类函数，以此来判断是使用relu激活函数还是sigmoid函数
    10、定义多层的整个神经网络的反向传播的计算函数
    11、更新参数
    12、执行预测函数
    13、模型的执行
"""


# 2初始化多层神经网络的参数
def initalize_parameters(layers_size):
    np.random.seed(3)
    m = len(layers_size)
    parameters = {}
    for i in range(1, m):
        parameters["W" + str(i)] = np.random.randn(layers_size[i], layers_size[i - 1]) / np.sqrt(layers_size[i - 1])
        parameters["b" + str(i)] = np.zeros((layers_size[i], 1))
    return parameters


# 4、定义向前计算函数
def liner_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


# 5、定义向前传播的分类函数，以此来判断是使用relu激活函数还是sigmoid函数
def liner_activation_forward(A_prev, W, b, activation):
    if activation == "relu":
        Z, liner_cache = liner_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        Z, liner_cache = liner_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    cache = (activation_cache, liner_cache)
    return A, cache


# 6、定义多层的整个神经网络的向前传播的计算函数
def L_model_forward(X, parameters):
    m = len(parameters) // 2  # 因为parameters里面包括W和b两个参数，因此除以2即为神经网络的层数
    A = X
    caches = []
    for i in range(1, m):  # 完成隐藏层的向前传播，不进行输出层的计算
        A_prev = A
        A, cache = liner_activation_forward(A_prev, parameters["W" + str(i)],
                                            parameters["b" + str(i)], "relu")
        caches.append(cache)

    # 完成最后一步输出层的向前传播计算
    AL, cache = liner_activation_forward(A, parameters["W" + str(m)], parameters["b" + str(m)], "sigmoid")
    caches.append(cache)

    return AL, caches


# 7、定义代价函数
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    return cost


# 8、定义反向传播的计算函数
def liner_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# 9、定义反向传播的分类函数，以此来判断是使用relu激活函数还是sigmoid函数
def liner_activation_backward(dA, cache, activation):
    activation_cache, liner_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = liner_backward(dZ, liner_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = liner_backward(dZ, liner_cache)

    return dA_prev, dW, db


# 10、定义多层的整个神经网络的反向传播的计算函数
def L_model_backward(AL, caches, Y):
    m = len(caches)
    grads = {}
    Y = Y.reshape(AL.shape)
    current_cache = caches[m - 1]
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA" + str(m)], grads["dW" + str(m)], grads["db" + str(m)] = liner_activation_backward(dAL, current_cache,
                                                                                                 "sigmoid")
    for i in reversed(range(m - 1)):
        current_cache = caches[i]
        dA_prev_temp, dW_temp, db_temp = liner_activation_backward(grads["dA" + str(i + 2)], current_cache, "relu")
        grads["dA" + str(i + 1)] = dA_prev_temp
        grads["dW" + str(i + 1)] = dW_temp
        grads["db" + str(i + 1)] = db_temp

    return grads


# 11、更新参数
def update_paremeters(parameters, grads, learning_rate):
    m = len(parameters) // 2
    for i in range(m):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - grads["dW" + str(i + 1)] * learning_rate
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - grads["db" + str(i + 1)] * learning_rate
    return parameters


# 12、执行预测函数
def prediction(parameters, X, Y):
    m = len(parameters) // 2  # 整除,神经网络的层数

    Y_prediction = np.zeros((1, X.shape[1]))
    AL, caches = L_model_forward(X, parameters)
    for i in range(0, AL.shape[1]):
        Y_prediction[0, i] = 1 if AL[0, i] > 0.5 else 0

    return Y_prediction


# 13、模型的执行
def L_model(train_x, train_y, test_x, test_y, iterator_num, learning_rate, layers_size):
    np.random.seed(1)
    costs = []
    parameters = initalize_parameters(layers_size)

    for i in range(iterator_num):
        AL, caches = L_model_forward(train_x, parameters)
        cost = compute_cost(AL, train_y)
        grads = L_model_backward(AL, caches, train_y)
        parameters = update_paremeters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print("第", i, "迭代代价为:", cost)
    plt.plot(np.squeeze(costs))
    plt.title("cost learning_rate:")
    plt.xlabel("iterator number:")
    plt.ylabel("cost:")
    plt.show()
    train_prediction = prediction(parameters, train_x, train_y)
    test_prediction = prediction(parameters, test_x, test_y)
    print("训练集准确性: ", format(100 - np.mean(np.abs(train_prediction - train_y)) * 100), "%")
    print("测试集准确性: ", format(100 - np.mean(np.abs(test_prediction - test_y)) * 100), "%")
    return parameters


# 1、加载数据
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
train_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
test_x_flatten = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

# 数据归一化,因为图片的像素是由r、g、b三原色组成，每个像素只能构成256中情况，
# 因此在数据进行归一化的时候需要除以255
train_set_x = train_x_flatten / 255
test_set_x = test_x_flatten / 255

# 3、定义神经网络结构
layers_size = [train_set_x.shape[0], 20, 7, 5, 1]
iterator_num = 2500
learning_rate = 0.0075

parameters = L_model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, iterator_num, learning_rate,
                     layers_size)


# my_image = "cat.jpg" # change this to the name of your image file
# my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
# ## END CODE HERE ##
#
# fname = "E:/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# # my_image = scipy.misc.imresize(image, size=(64,64)).reshape((64*64*3,1))
# # my_predicted_image = prediction(my_image, my_label_y, parameters)
#
# plt.imshow(image)
# #print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
