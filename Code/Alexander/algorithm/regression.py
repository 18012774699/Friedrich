# 模型训练
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler


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


# 岭回归
def ridge_regression():
    # 模拟数据
    X = 2 * np.random.rand(1000, 1)
    y = 4 + 3 * X + np.random.randn(1000, 1)

    # 岭回归的封闭方程的解
    ridge_reg = Ridge(alpha=1, solver="cholesky")
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))

    # 使用随机梯度法进行求解
    sgd_reg = SGDRegressor(penalty="l2")
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.predict([[1.5]]))


# 早期停止法
def early_stopping():
    # 模拟数据
    m = 100
    X = 6 * np.random.rand(m, 1) - 3  # [-3,3)
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_train_poly_scaled, X_val_poly_scaled, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)

    # 当warm_start=True时，调用fit()方法后，训练会从停下来的地方继续，而不是从头重新开始。
    sgd_reg = SGDRegressor(max_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)

    minimum_val_error = float("inf")
    best_epoch = None
    best_model = None
    for epoch in range(1000):
        sgd_reg.fit(X_train_poly_scaled, y_train.ravel())
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        val_error = mean_squared_error(y_val_predict, y_val)
        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)
    return best_model, best_epoch


if __name__ == "__main__":
    # 线性回归
    # linear_regression()

    # 多项式回归
    # polynomial_regression()

    # 岭回归
    # ridge_regression()

    # 早期停止法
    best_model, best_epoch = early_stopping()
    print(best_epoch, best_model)
