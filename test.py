#机器学习线性判别分析3.5题
import numpy as np
import matplotlib.pyplot as plt
data = [[0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]]   #书中89页西瓜数据集
#数据集按瓜好坏分类
data = np.array([i[:-1] for i in data])
X0 = np.array(data[:8])
X1 = np.array(data[8:])
print(X0)
#求正反例均值
miu0 = np.mean(X0, axis=0).reshape((-1, 1))
miu1 = np.mean(X1, axis=0).reshape((-1, 1))
print(miu0)
#求协方差
cov0 = np.cov(X0, rowvar=False)
cov1 = np.cov(X1, rowvar=False)
print(cov0)
#求出w
S_w = np.mat(cov0 + cov1)
Omiga = S_w.I * (miu0 - miu1)
#画出点、直线
plt.scatter(X0[:, 0], X0[:, 1], c='b', label='+', marker = '+')
plt.scatter(X1[:, 0], X1[:, 1], c='r', label='-', marker = '_')
plt.plot([0, 1], [0, -Omiga[0] / Omiga[1]], label='y')
plt.xlabel('密度', fontproperties='SimHei', fontsize=15, color='green');
plt.ylabel('含糖率', fontproperties='SimHei', fontsize=15, color='green');
plt.title(r'3.5 线性判别分析', fontproperties='SimHei', fontsize=25);
plt.legend()
plt.show()