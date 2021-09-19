# -*- coding: utf-8 -*-
# 对率回归分类
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
x = np.array([[0.697,0.774,0.634,0.608,0.556,0.430,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719],
             [0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def newton():
    # 定义初始值,分别对应密度、含糖率与常数项的系数
    beta = np.array([[0], [0], [1]])  # β列向量
    n = 0
    epsilon = 0.000001
    while 1:
        # 对β进行转置取第一行
        # 再与x相乘（dot）,beta_x表示β转置乘以x)
        beta_x = np.dot(beta.T[0], x)
        # 先求关于β的一阶导数和二阶导数
        dbeta = 0
        d2beta = 0
        for i in range(17):
            # 一阶导数
            dbeta = dbeta - np.dot(np.array([x[:, i]]).T,
                                   (y[i] - (np.exp(beta_x[i]) / (1 + np.exp(beta_x[i])))))
            # 二阶导数
            d2beta = d2beta + np.dot(np.array([x[:, i]]).T, np.array([x[:, i]]).T.T) * (
                    np.exp(beta_x[i]) / (1 + np.exp(beta_x[i]))) * (
                             1 - (np.exp(beta_x[i]) / (1 + np.exp(beta_x[i]))))
        # 得到牛顿方向
        d = - np.dot(linalg.inv(d2beta), dbeta)
        # 迭代终止条件
        if np.linalg.norm(d) <= epsilon:
            break  # 满足条件直接跳出循环
        # 牛顿迭代法更新β
        beta = beta + d
        n = n + 1
    print('密度的系数为:  ', beta[0], '\n','含糖率的系数为:', beta[1], '\n','常数项为:     ', beta[2])
    print('迭代次数：', n)
    return beta

def show(beta):
    plt.title('Logistic Regression')
    plt.xlabel('density'.title())
    plt.ylabel('sugar rate'.title())
    # 绘制正例
    ps = plt.scatter(x[0][0:8], x[1][0:8], c='r', marker='*')
    # 绘制反例
    ns = plt.scatter(x[0][8:], x[1][8:], c='b', marker='o')
    plt.legend((ps, ns), ('positive sample', 'negative sample'))
    # 计算分界线上的点：w1*x1+w2*x2+b=0
    x1 = np.arange(0, 1, 0.01)
    x2 = (- x1 * beta[0] - beta[2]) / beta[1]
    # 绘制分界线
    plt.plot(x1, x2)
    plt.show()

if __name__ == '__main__':
    beta=newton()
    show(beta)