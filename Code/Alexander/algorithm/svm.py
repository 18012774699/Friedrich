# SVM 是严格的二分类器。
# SVM 分类器适用于小的训练集，OvO 策略

# 较小的C会导致更宽的“街道”，但更多的间隔违规。
# 如果你的 SVM 模型过拟合，你可以尝试通过减小超参数C去调整。
# 不同于 Logistic 回归分类器，SVM 分类器不会输出每个类别的概率。
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

svm_clf = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])
svm_clf.fit(X, y)
print(svm_clf.predict([[5.5, 1.7]]))

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])

polynomial_svm_clf.fit(X, y)
