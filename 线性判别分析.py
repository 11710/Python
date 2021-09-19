# -*- coding: utf-8 -*-
#机器学习线性判别分析
import numpy as np
import matplotlib.pyplot as plt
#数据集按瓜好坏分类
Data = np.array([[0.697,0.774,0.634,0.608,0.556,0.430,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719],
             [0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103]])
#求正反例均值
x0 = np.array([i[0:8] for i in Data]).T
x1 = np.array([i[8:] for i in Data]).T

miu0 = np.mean(x0, axis=0).reshape((-1, 1))
miu1 = np.mean(x1, axis=0).reshape((-1, 1))
print("u0的均值为：\n",miu0)
print("u1的均值为：\n",miu1)
#求协方差
cov0 = np.cov(x0, rowvar=False)
cov1 = np.cov(x1, rowvar=False)
#求出w,转换为矩阵
S_w = np.mat(cov0 + cov1)
print("类内散度矩阵为: \n",S_w)
Omiga = S_w.I * (miu0 - miu1)
#画出点、直线
print("w的值为：\n",Omiga)
plt.title('Linear Discriminant Analysis')
plt.xlabel('density'.title())
plt.ylabel('sugar rate'.title())
ps = plt.scatter(x0[:, 0], x0[:, 1], c='b', marker = '*')
ns = plt.scatter(x1[:, 0], x1[:, 1], c='r', marker = 'o')
plt.legend((ps, ns), ('positive sample', 'negative sample'))
plt.plot([0, 1], [0, -Omiga[0] / Omiga[1]])
# 绘制分界线
plt.show()