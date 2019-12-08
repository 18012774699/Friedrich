# 使用sklearn，求解析解
import numpy as np
from sklearn.linear_model import LinearRegression

# 随机样本因素，返回100行1列随机数x，范围[0, 2)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
# 前者为w0，后者为剩余的w1、w2、w3...，这里只有w1
print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.array([[0], [2]])
print(lin_reg.predict(X_new))
