# SVM 是严格的二分类器。
# SVM 分类器适用于小的训练集，OvO 策略
# 不同于 Logistic 回归分类器，SVM 分类器不会输出每个类别的概率
#
# 较小的C会导致更宽的“街道”，但更多的间隔违规
# 如果你的 SVM 模型过拟合，你可以尝试通过减小超参数C去调整
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC


# 线性支持向量机
def linear_svc():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

    svm_clf = Pipeline([("scaler", StandardScaler()),
                        ("linear_svc", LinearSVC(C=1, loss="hinge"))])
    svm_clf.fit(X, y)
    print(svm_clf.predict([[5.5, 1.7]]))


# 非线性支持向量机
def nonlinear_svc(X, y):
    polynomial_svm_clf = Pipeline([("poly_features", PolynomialFeatures(degree=3)),
                                   ("scaler", StandardScaler()),
                                   ("svm_clf", LinearSVC(C=10, loss="hinge"))])
    polynomial_svm_clf.fit(X, y)


# 多项式核
# “核技巧”（kernel trick）的神奇数学技巧，并没有增加任何特征
# 它可以取得就像你添加了许多多项式，甚至有高次数的多项式，一样好的结果。
def polynomial_kernel(X, y):
    poly_kernel_svm_clf = Pipeline([("scaler", StandardScaler()),
                                    # 3阶，超参数coef0控制了高阶多项式与低阶多项式对模型的影响
                                    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))])
    poly_kernel_svm_clf.fit(X, y)


# 高斯RBF核，相似特征法
def gaussian_rbf_kernel(X, y):
    rbf_kernel_svm_clf = Pipeline([("scaler", StandardScaler()),
                                   # 超参数gamma(γ)
                                   # 如果你的模型过拟合，你应该减小γ值，若欠拟合，则增大γ（与超参数C相似）。
                                   ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))])
    rbf_kernel_svm_clf.fit(X, y)


if __name__ == "__main__":
    # linear_svc()

    # 生成月亮数据集
    X, y = make_moons(n_samples=1000, noise=0.1)

    nonlinear_svc(X, y)

    polynomial_kernel(X, y)

    gaussian_rbf_kernel(X, y)
