# 多项式回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3    # [-3,3)
# y = 0.5X^2 + X + 2 + error
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, 'b.')    # blue point

d = {1: 'g-', 2: 'r+', 10: 'y*'}
for i in d:
    # 多项式回归，degree = 次方数
    poly_features = PolynomialFeatures(degree=i, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    print(X[0])             # 第一个样本的特征
    print(X_poly[0])        # 第一个样本特征的i阶多项式
    # X[:,0]就是取所有行的第0个数据
    print(X_poly[:, 0])     # 所有样本的特征
    print("--------------------")

    lin_reg = LinearRegression(fit_intercept=True)
    # 建模
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    # 预测
    y_predict = lin_reg.predict(X_poly)
    plt.plot(X_poly[:, 0], y_predict, d[i])
    print("======================")

plt.show()
