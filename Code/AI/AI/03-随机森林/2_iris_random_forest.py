# 随机森林：多次求解，少数服从多数
# 随机森立本身就是一种Bagging的思路
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]  # 花萼长度和宽度
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 以下2种方式等价
# n_estimators：决策树个数，max_leaf_nodes：复杂程度，n_jobs：线程数
rnd_clf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=16, n_jobs=1)
rnd_clf.fit(X_train, y_train)

# splitter：每棵树随机的维度，max_samples：每棵树随机的样本比例
bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
    n_estimators=15, max_samples=1.0, bootstrap=True, n_jobs=1)
bag_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
y_pred_bag = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))
print(accuracy_score(y_test, y_pred_bag))

# Feature Importance
# 相关性
# 1.Pearson相关系数
# 2.L1正则回归
# 3.树的特征分裂顺序
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)
