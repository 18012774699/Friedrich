# 随机梯度下降
import numpy as np

X0 = np.ones((100, 1))
# 随机样本因素
X1 = 2 * np.random.rand(100, 1)
y = 4 + 3 * X1 + np.random.randn(100, 1)
X = np.c_[X0, X1]
print(X)

n_epochs = 500
t0, t1 = 5, 50  # 超参数
m = 100


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)
# range[0,x)
for epoch in range(n_epochs):   # 外层500轮次，总计50000次
    for i in range(m):
        # 返回[0, m)随机整数
        random_index = np.random.randint(m)
        # 提取对应的随机样本
        xi = X[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        # 不断减小学习率
        learning_rate = learning_schedule(epoch * m + i)
        theta = theta - learning_rate * gradients

print(theta)
