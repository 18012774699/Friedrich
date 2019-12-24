# SVM 回归
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR


def linear_svr(X, y):
    svm_reg = Pipeline([("scaler", StandardScaler()),
                        ("linear_svc", LinearSVR(epsilon=1.5))])
    svm_reg.fit(X, y.ravel())
    print(svm_reg.predict([[5.5, 1.7]]))


if __name__ == "__main__":
    # 模拟数据
    X = 2 * np.random.rand(1000, 1)
    y = 4 + 3 * X + np.random.randn(1000, 1)

    linear_svr(X, y)
