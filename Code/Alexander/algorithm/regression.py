# 模型训练
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# 正态方程
def normal_equation(X, y):
    print("normal_equation:")
    X_b = np.c_[np.ones((len(X), 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(theta_best)

    # 使用θ hat来进行预测
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(theta_best)
    print(y_predict)

    # 同上
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    print(lin_reg.predict(X_new))

    # 画出这个模型的图像
    plt.plot(X_new, y_predict, "r-")
    plt.plot(X, y, "b.")
    plt.axis([0, 2, 0, 15])
    plt.show()


# 批量梯度下降
def batch_gradient_descent(X, y, eta: float = 0.1, n_iterations: int = 1000):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(2, 1)  # 随机初始值

    for iteration in range(n_iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
    print("batch_gradient_descent:")
    print(theta)


# 随机梯度下降
def stochastic_gradient_descent(X, y):
    sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
    sgd_reg.fit(X, y.ravel())
    print("stochastic_gradient_descent:")
    print(sgd_reg.intercept_, sgd_reg.coef_)


# 线性回归
def linear_regression():
    # 模拟数据
    X = 2 * np.random.rand(1000, 1)
    y = 4 + 3 * X + np.random.randn(1000, 1)

    # 正态方程
    normal_equation(X, y)

    # 批量梯度下降
    batch_gradient_descent(X, y)

    # 随机梯度下降
    stochastic_gradient_descent(X, y)


# 绘制学习曲线,画出以训练集规模为自变量的训练集函数。
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.title("mean_squared_error")
    plt.legend()
    plt.show()


# 多项式回归
def polynomial_regression():
    # 模拟数据
    m = 100
    X = 6 * np.random.rand(m, 1) - 3  # [-3,3)
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

    d = {1: 'r-', 2: 'bo', 300: 'g+'}
    for key, value in d.items():
        # 多项式回归，degree = 次方数
        poly_features = PolynomialFeatures(degree=key, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        print(X[0])  # 第一个样本的特征
        print(X_poly[0])  # 第一个样本特征的i阶多项式
        # X[:,0]就是取所有行的第0个数据
        print(X_poly[:, 0])  # 所有样本的特征
        print("--------------------")

        lin_reg = LinearRegression()
        # 建模
        lin_reg.fit(X_poly, y)
        print(lin_reg.intercept_, lin_reg.coef_)

        # 预测
        y_predict = lin_reg.predict(X_poly)
        plt.plot(X, y_predict, value)
        print("======================")
    plt.plot(X, y, 'b.')
    plt.show()

    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)


if __name__ == "__main__":
    # 线性回归
    # linear_regression()

    # 多项式回归
    polynomial_regression()


