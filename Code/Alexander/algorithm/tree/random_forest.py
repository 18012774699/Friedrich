from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


# 随机森林
def random_forest():
    # 生成月亮数据集
    X, y = make_moons(n_samples=1000, noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    # 大致相当于之前的RandomForestClassifier
    bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16), n_estimators=500,
                                max_samples=1.0, bootstrap=True, n_jobs=-1)

    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred_rf))


# 特征重要度
def feature_importances():
    iris = load_iris()
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rnd_clf.fit(iris["data"], iris["target"])
    for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)


if __name__ == "__main__":
    random_forest()

    feature_importances()
