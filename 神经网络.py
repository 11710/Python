import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 数据预处理
def preprocess(data):
    # 将非数映射数字
    for title in data.columns:
        if data[title].dtype == 'object':
            #是对不连续的数字或者文本进行编号
            encoder = LabelEncoder()
            data[title] = encoder.fit_transform(data[title])
            # 去均值和方差归一化
    # 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
    ss = StandardScaler()
    X = data.drop('好瓜', axis=1)
    Y = data['好瓜']
    X = ss.fit_transform(X)
    x, y = np.array(X), np.array(Y).reshape(Y.shape[0], 1)
    return x, y

# 定义Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 求导
def d_sigmoid(x):
    return x * (1 - x)

# 累积BP算法
def accumulate_BP(x, y, dim=10, eta=0.8, max_iter=500):
    n_samples = x.shape[0]
    w1 = np.zeros((x.shape[1], dim))
    b1 = np.zeros((n_samples, dim))
    w2 = np.zeros((dim, 1))
    b2 = np.zeros((n_samples, 1))
    losslist = []
    for ite in range(max_iter):
        ##前向传播
        u1 = np.dot(x, w1) + b1
        out1 = sigmoid(u1)
        u2 = np.dot(out1, w2) + b2
        out2 = sigmoid(u2)
        loss = np.mean(np.square(y - out2)) / 2
        losslist.append(loss)
        print('iter:%d  loss:%.4f' % (ite, loss))
        ##更新
        d_out2 = -(y - out2)
        d_u2 = d_out2 * d_sigmoid(out2)
        d_w2 = np.dot(np.transpose(out1), d_u2)
        d_b2 = d_u2
        d_out1 = np.dot(d_u2, np.transpose(w2))
        d_u1 = d_out1 * d_sigmoid(out1)
        d_w1 = np.dot(np.transpose(x), d_u1)
        d_b1 = d_u1

        w1 = w1 - eta * d_w1
        w2 = w2 - eta * d_w2
        b1 = b1 - eta * d_b1
        b2 = b2 - eta * d_b2

    ##补充Loss可视化代码
    plt.figure()
    plt.plot([i + 1 for i in range(max_iter)], losslist)
    plt.legend(['accumlated BP'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    return w1, w2, b1, b2

# 标准BP算法
def standard_BP(x, y, dim=10, eta=0.8, max_iter=500):
    n_samples = 1
    w1 = np.zeros((x.shape[1], dim))
    b1 = np.zeros((n_samples, dim))
    w2 = np.zeros((dim, 1))
    b2 = np.zeros((n_samples, 1))
    losslist = []
    # 补充标准BP算法代码
    for ite in range(max_iter):
        loss_per_ite = []
        for m in range(x.shape[0]):
            xi, yi = x[m, :], y[m, :]
            xi, yi = xi.reshape(1, xi.shape[0]), yi.reshape(1, yi.shape[0])
            # 前向传播
            u1 = np.dot(xi, w1) + b1
            out1 = sigmoid(u1)
            u2 = np.dot(out1, w2) + b2
            out2 = sigmoid(u2)
            loss = np.square(yi - out2) / 2
            loss_per_ite.append(loss)
            #print('iter:%d  loss:%.4f' % (ite, loss))
            # 反向传播
            d_out2 = -(yi - out2)
            d_u2 = d_out2 * d_sigmoid(out2)
            d_w2 = np.dot(np.transpose(out1), d_u2)
            d_b2 = d_u2
            d_out1 = np.dot(d_u2, np.transpose(w2))
            d_u1 = d_out1 * d_sigmoid(out1)
            d_w1 = np.dot(np.transpose(xi), d_u1)
            d_b1 = d_u1

            w1 = w1 - eta * d_w1
            w2 = w2 - eta * d_w2
            b1 = b1 - eta * d_b1
            b2 = b2 - eta * d_b2
        losslist.append(np.mean(loss_per_ite))

    # 补充Loss可视化代码
    plt.figure()
    plt.plot([i + 1 for i in range(max_iter)], losslist)
    plt.legend(['standard BP'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

    return w1, w2, b1, b2

# 测试

def main():
    data = pd.read_table('watermelon.txt', delimiter=',')
    #删除编号第一列
    data.drop('编号', axis=1, inplace=True)
    # 将非数映射数字
    x, y = preprocess(data)
    dim = 10
    w1, w2, b1, b2 = standard_BP(x, y, dim)
    # w1,w2,b1,b2 = accumulate_BP(x,y,dim)

    u1 = np.dot(x, w1) + b1
    out1 = sigmoid(u1)
    u2 = np.dot(out1, w2) + b2
    out2 = sigmoid(u2)
    y_pred = np.round(out2)

    result = pd.DataFrame(np.hstack((y, y_pred)), columns=['真值', '预测'])
    print(result)

# 补充测试代码，根据当前的x，预测其类别；
if __name__ == '__main__':
    main()

