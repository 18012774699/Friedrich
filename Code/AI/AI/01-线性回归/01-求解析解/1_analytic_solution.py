# 求解析解
# X为样本因素，Y为样本结果
import numpy as np
import matplotlib.pyplot as plt

# 这里相当于是随机X维度X1，rand是随机均匀分布,over[0, 1)
# 随机样本因素，返回100行1列随机数x，范围[0, 2)
X1 = 2 * np.random.rand(100, 1)

# 人为的创造一列'真实数据'Y，用np.random.(100, 1)设置误差error
# randn是标准正态分布, μ=0，方差=1
# 下面符合公式：y = w0*x0 + w1*x1 + error ，其中w0=4,w1=3,x0恒为1
y = 4 + 3 * X1 + np.random.randn(100, 1)

# 整合X0和X1
# X0恒为1，所以用ones表示100行1列的数字1, 列向量
X0 = np.ones((100, 1))
# c_对应combine，整合
# 最终结果，X每行都为一个样本，包含所有因素(x0, x1)
X = np.c_[X0, X1]
print(X)

# 常规等式求解θ，公式θ=(X^T * X)^-1 * X^T * y
# inv求逆，T转置
theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print(theta_best)

#########################
print("####################")
# 创建测试集里面的X1
X0_new = np.ones((2, 1))        # 2行1列的1, 列向量
X1_new = np.array([[0], [2]])   # 二维列表转二维数组
X_new = np.c_[X0_new, X1_new]   # 两行新样本
print(X_new)
# y^= X*θ
y_predict = X_new.dot(theta_best)
print(y_predict)

# 绘制图形
plt.plot(X1_new, y_predict, 'r-')   # 两组预测值连线
plt.plot(X1, y, 'b.')               # 100组实际值取点
plt.axis([0, 2, 0, 15])             # x, y轴区间
plt.show()
