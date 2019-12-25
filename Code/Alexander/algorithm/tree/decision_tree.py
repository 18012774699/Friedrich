from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# 你可以通过使用export_graphviz()方法，通过生成一个叫做iris_tree.dot的图形定义文件将一个训练好的决策树模型可视化。
export_graphviz(tree_clf, out_file="iris_tree.dot", feature_names=iris.feature_names[2:], class_names=iris.target_names,
                rounded=True, filled=True)

tree_clf.predict_proba([[5, 1.5]])
print(tree_clf.predict([[5, 1.5]]))

