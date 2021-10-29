from sklearn import svm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def plt_support_(clf, X_, y_, kernel, c):
    pos = y_ == 1
    neg = y_ == -1
    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(0, 0.8, 600)
    #从坐标向量中返回坐标矩阵
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    #将多维数据转换成一维数据
    Z_rbf = clf.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)
    cs = plt.contour(X_tmp, Y_tmp, Z_rbf, [0], colors='orange', linewidths=1)
    #plt.clabel(cs, fmt={cs.levels[0]: 'decision boundary'})
    plt.scatter(X_[pos, 0], X_[pos, 1], label='1', color='r')
    plt.scatter(X_[neg, 0], X_[neg, 1], label='0', color='b')
    plt.scatter(X_[clf.support_, 0], X_[clf.support_, 1], color='none',marker='o', edgecolors='g', s=200,
                label='support_vectors')
    # 将字符串标注到图中
    plt.legend()
    plt.title('{} kernel, C={}'.format(kernel, c))
    plt.show()
path = r'watermelon3_0a_Ch.txt'
data = pd.read_table(path, delimiter=' ', dtype=float)

X = data.iloc[:, [0, 1]].values
y = data.iloc[:, 2].values

y[y == 0] = -1
C = 1000

clf_rbf = svm.SVC(C=C, kernel='rbf')
clf_rbf.fit(X, y.astype(int))
print('高斯核：')
print('预测值：', clf_rbf.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_rbf.support_)

print('-' * 40)
clf_linear = svm.SVC(C=C, kernel='linear')
clf_linear.fit(X, y.astype(int))
print('线性核：')
print('预测值：', clf_linear.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_linear.support_)

plt_support_(clf_rbf, X, y, 'rbf', C)
plt_support_(clf_linear, X, y, 'linear', C)
