import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

HOUSING_PATH = "../../datasets/housing"


def pca(X):
    X_centered = X - X.mean(axis=0)
    U, s, V = np.linalg.svd(X_centered)
    # 提取前两个主成分 PC
    c1 = V.T[:, 0]
    c2 = V.T[:, 1]
    # 投影到d维空间
    W2 = V.T[:, :2]
    X2D = X_centered.dot(W2)


def pca_sklearn(X):
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(X)
    print(X2D)
    # print(pca.components_.T[:, 0])

    # 方差解释率
    print(pca.explained_variance_ratio_)

    # 选择正确的维度
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)


if __name__ == "__main__":
    housing = pd.read_csv(os.path.join(HOUSING_PATH, "housing.csv"))
    X = housing.drop(["median_house_value", "ocean_proximity"], axis=1)
    X = X[:5]

    # pca(X)

    pca_sklearn(X)
