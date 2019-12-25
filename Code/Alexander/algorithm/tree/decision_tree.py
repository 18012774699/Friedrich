# 决策树的众多特性之一就是， 它不需要太多的数据预处理， 尤其是不需要进行特征的缩放或者归一化。
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor


def classification():
    iris = load_iris()
    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)

    # 你可以通过使用export_graphviz()方法，通过生成一个叫做iris_tree.dot的图形定义文件将一个训练好的决策树模型可视化。
    export_graphviz(tree_clf, out_file="iris_tree.dot", feature_names=iris.feature_names[2:],
                    class_names=iris.target_names,
                    rounded=True, filled=True)

    tree_clf.predict_proba([[5, 1.5]])
    print(tree_clf.predict([[5, 1.5]]))


def regression():
    # 模拟数据
    m = 100
    X = 6 * np.random.rand(m, 1) - 3  # [-3,3)
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

    tree_reg = DecisionTreeRegressor(max_depth=2)
    tree_reg.fit(X, y)


if __name__ == "__main__":
    classification()

    regression()
