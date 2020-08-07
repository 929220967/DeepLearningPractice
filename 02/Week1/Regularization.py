import numpy as np
from reg_utils import *
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# loss函数，带L2正则
def compute_cost_with_regularization(AL, Y, params, lamdb):  # lamdb:是设置的λ
    m = Y.shape[1]
    W1 = params["W1"]
    W2 = params["W2"]
    W3 = params["W3"]

    cross_entropy_cost = compute_cost(AL, Y)

    # 根据公式计算L2正则化
    L2_cost = lamdb * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)
    loss = cross_entropy_cost + L2_cost

    return loss


# 向前传播，带dropout
def forward_propagation(X, params, keep_prob):
    np.random.seed(1)

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    # 随机失活
    D1 = np.random.randn(A1.shape[0], A1.shape[1]) < keep_prob
    A1 = A1 * D1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    D2 = np.random.randn(A2.shape[0], A2.shape[1]) < keep_prob
    A2 = A2 * D2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    AL = sigmoid(Z3)

    cache = (Z1, Z2, Z3, A1, A2, AL, W1, W2, W3, b1, b2, b3, D1, D2)

    return AL, cache


# 向后传播
def backward_propagation(X, Y, cache, lamdb, keep_prob):
    m = X.shape[1]
    Z1, Z2, Z3, A1, A2, AL, W1, W2, W3, b1, b2, b3, D1, D2 = cache
    dZ3 = AL - Y

    dW3 = (1 / m) * (np.dot(dZ3, A2.T) + (lamdb * W3))
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2 / keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * (np.dot(dZ2, A1.T) + (lamdb * W2))
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1 / keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * (np.dot(dZ1, X.T) + (lamdb * W1))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dZ3": dZ3,
        "dW3": dW3,
        "db3": db3,
        "dA2": dA2,
        "dZ2": dZ2,
        "dW2": dW2,
        "db2": db2,
        "dA1": dA1,
        "dZ1": dZ1,
        "dW1": dW1,
        "db1": db1,
    }
    return grads


# 定义模型
def model(train_X, train_Y, learning_rate, iterator_num, keep_prob=1, lamdb=0):
    grads = {}
    costs = []
    m = train_X.shape[1]
    layer_dims = [train_X.shape[0], 20, 3, 1]

    params = initialize_parameters(layer_dims)

    for i in range(iterator_num):
        AL, cache = forward_propagation(train_X, params, keep_prob)
        cost = compute_cost_with_regularization(AL, train_Y, params, lamdb)
        grads = backward_propagation(train_X, train_Y, cache, lamdb, keep_prob)
        params = update_parameters(params, grads, learning_rate)

        if i % 100 == 0:
            print("第", i, "次迭代，代价为:", cost)
        costs.append(cost)

    plt.rcParams["figure.figsize"] = (15.0, 4.0)
    plt.subplot(1, 2, 1)
    plt.plot(costs)
    plt.title("lamdb: " + str(lamdb) + " keep_prob: " + str(keep_prob))
    plt.xlabel("iterator_num")
    plt.ylabel("cost")

    plt.subplot(1, 2, 2)
    plt.title("lamdb: " + str(lamdb) + " keep_prob: " + str(keep_prob))
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, np.squeeze(train_Y))

    return params


# 定义预测
def prediction(X, Y, params, keep_prob):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    AL, cache = forward_propagation(X, params, keep_prob)
    for i in range(m):
        Y_prediction[0, i] = 1 if AL[0, i] > 0.5 else 0

    return Y_prediction


# 加载数据
train_X, train_Y, test_X, test_Y = load_2D_dataset()
params = model(train_X, train_Y, learning_rate=0.03, iterator_num=30000, keep_prob=0.81, lamdb=0.76)
Y_prediction = prediction(test_X, test_Y, params, keep_prob=0.86)
print(Y_prediction)
