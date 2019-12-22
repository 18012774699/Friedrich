import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# 逻辑回归
def logistic_regression(iris):
    X = iris["data"][:, 3:]  # 花瓣宽度
    y = (iris["target"] == 2).astype(np.int)

    log_reg = LogisticRegression()
    log_reg.fit(X, y)

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
    plt.show()

    print(log_reg.predict([[1.7], [1.5]]))


# Softmax 回归
def softmax_regression(iris):
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]

    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
    softmax_reg.fit(X, y)
    print(softmax_reg.predict([[5, 2]]))
    print(softmax_reg.predict_proba([[5, 2]]))


if __name__ == "__main__":
    iris = datasets.load_iris()
    print(list(iris.keys()))

    # 逻辑回归
    logistic_regression(iris)

    # Softmax 回归
    softmax_regression(iris)

