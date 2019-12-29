from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# 适应性提升
def Adaboost(X_train, X_test, y_train, y_test):
    # 如果你的 Adaboost 集成过拟合了训练集，你可以尝试减少基分类器的数量或者对基分类器使用更强的正则化。
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=200, algorithm="SAMME.R",
                                 learning_rate=0.5)
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))


# 梯度提升原理
def gradient_boosting(X_train, X_test, y_train, y_test):
    # 这个方法是去使用新的分类器去拟合前面分类器预测的残差
    tree_reg1 = DecisionTreeRegressor(max_depth=2)
    tree_reg1.fit(X_train, y_train)

    y2 = y_train - tree_reg1.predict(X_train)
    tree_reg2 = DecisionTreeRegressor(max_depth=2)
    tree_reg2.fit(X_train, y2)

    y3 = y2 - tree_reg1.predict(X_train)
    tree_reg3 = DecisionTreeRegressor(max_depth=2)
    tree_reg3.fit(X_train, y3)

    y_pred = sum(tree.predict(X_test) for tree in (tree_reg1, tree_reg2, tree_reg3))

    # GBRT, 同上
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
    gbrt.fit(X, y)


# 梯度提升早停技术
def gradient_boosting_early_stopping(X_train, X_test, y_train, y_test):
    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_test)
        val_error = mean_squared_error(y_test, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                # print(n_estimators)
                break  # early stopping


if __name__ == "__main__":
    # 生成月亮数据集
    X, y = make_moons(n_samples=1000, noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    Adaboost(X_train, X_test, y_train, y_test)

    gradient_boosting(X_train, X_test, y_train, y_test)

    gradient_boosting_early_stopping(X_train, X_test, y_train, y_test)
