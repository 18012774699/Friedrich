# 岭回归
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# auto: 自动选择解析解或梯度下降法
# sag:  SGD
ridge_reg = Ridge(alpha=1, solver='sag')
ridge_reg.fit(X, y)
print(ridge_reg.predict(1.5))       # 预测1.5
print("W0 =", ridge_reg.intercept_) # 截距W0
print("W1 =", ridge_reg.coef_)      # W1...Wn
print("========================")

# SGDRegressor(penalty='l2')等价于Ridge，使用L2正则
sgd_reg = SGDRegressor(penalty='l2', max_iter=1000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict(1.5))
print("W0 =", sgd_reg.intercept_)
print("W1 =", sgd_reg.coef_)
