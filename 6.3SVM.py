#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
#data_set = load_iris()
data_set = load_breast_cancer()
X = data_set.data  # feature
feature_names = data_set.feature_names
y = data_set.target  # label
target_names = data_set.target_names

# 画散点图
import matplotlib.pyplot as plt
f1 = plt.figure(1)
p1 = plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r', label=target_names[0])  # feature
p2 = plt.scatter(X[y == 1, 0], X[y == 1, 1], color='g', label=target_names[1])  # feature
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("dataset")
plt.legend(loc='upper right')
plt.grid(True, linewidth=0.3)

plt.show()

# 数据规范化
from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)

# 模型拟合和测试
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

# 训练集和测试集的生成
X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.5, random_state=0)

# 模型拟合，测试，可视化
# 基于线性内核和高斯内核
for fig_num, kernel in enumerate(('linear', 'rbf')):
    accuracy = []
    c = []
    for C in range(1, 1000, 1):
        # 初始化
        clf = svm.SVC(C=C, kernel=kernel)
        # 训练
        clf.fit(X_train, y_train)
        # 测试
        y_pred = clf.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        c.append(C)
    print('max accuracy of %s kernel SVM: %.3f' % (kernel, max(accuracy)))
    # 绘制准确率
    f2 = plt.figure(2)
    plt.plot(c, accuracy)
    plt.title('{} kernel'.format(kernel))
    # 惩罚参数
    plt.xlabel('penalty parameter')
    # 精确度
    plt.ylabel('accuracy')
    plt.show()

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
mlp=MLPClassifier(random_state=0,max_iter=1000,alpha=1).fit(X_train,y_train)
print("Training set score(mlp): {:.3f}".format(mlp.score(X_train, y_train)))
print("Test set score(mlp): {:.3f}".format(mlp.score(X_test, y_test)))

tree = DecisionTreeClassifier(random_state=0,max_depth=4).fit(X_train,y_train)
print("Accuracy on training set(tree):{:3f}".format(tree.score(X_train,y_train)))
print("Accuracy on test set(tree):{:3f}".format(tree.score(X_test,y_test)))