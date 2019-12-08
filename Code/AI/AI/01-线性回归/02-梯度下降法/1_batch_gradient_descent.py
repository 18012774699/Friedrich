# 批量梯度下降法
# X为样本因素，Y为样本结果
import numpy as np

X0 = np.ones((100, 1))
# 随机样本因素，返回100行1列随机数x，范围[0, 2)
X1 = 2 * np.random.rand(100, 1)
y = 4 + 3 * X1 + np.random.randn(100, 1)
X = np.c_[X0, X1]
# print(X)

learning_rate = 0.1     # 学习率
n_iterations = 10000    # 迭代次数
m = 100                 # 100个样本

# 1，初始化theta，w0...wn
theta = np.random.randn(2, 1)
count = 0

# 4，不会直接设置阈值超参数，而是迭代次数到了，我们就认为收敛了
for iteration in range(n_iterations):
    count += 1
    # 2，接着求梯度gradient（包括多个维度g0,g1...gn）
    # 除以m，是为了随着样本数量增多而均摊损失
    gradients = 1 / m * X.T.dot(X.dot(theta) - y)
    # 3，应用公式调整theta值
    theta = theta - learning_rate * gradients

print(count)
print(theta)

