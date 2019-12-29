# SVM 回归
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.svm import SVR


def linear_svr(X, y):
    svm_reg = Pipeline([("scaler", StandardScaler()),
                        # “街道”的宽度由超参数ϵ控制
                        ("linear_svc", LinearSVR(epsilon=1.5))])
    svm_reg.fit(X, y.ravel())
    print(svm_reg.predict(X))


if __name__ == "__main__":
    # 模拟数据
    X = 2 * np.random.rand(1000, 1)
    y = 4 + 3 * X + np.random.randn(1000, 1)

    linear_svr(X, y)

    svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg.fit(X, y)
