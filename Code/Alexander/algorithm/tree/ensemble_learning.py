from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# 软投票和硬投票
def hard_and_sort_vote(X_train, X_test, y_train, y_test):
    log_clf = LogisticRegression(solver='liblinear')
    dtc_clf = DecisionTreeClassifier()
    svm_clf = SVC(gamma="scale", probability=True)
    voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('dt', dtc_clf), ('svc', svm_clf)], voting='soft')
    # voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('dt', dtc_clf), ('svc', svm_clf)], voting='hard')
    voting_clf.fit(X_train, y_train)

    for clf in (log_clf, dtc_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# Bagging 和 Pasting
def bag_and_paste(X_train, X_test, y_train, y_test):
    # n_estimators: 训练个数
    # max_samples: 采样个数
    # bootstrap: True/False, Bagging/Pasting
    # n_jobs: -1 代表着 sklearn 会使用所有空闲核
    # oob_score=True: 开启Out-of-Bag评估
    # BaggingClassifier也支持采样特征。它被两个超参数max_features和bootstrap_features控制。
    bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                                bootstrap=True, oob_score=True, n_jobs=-1)
    bag_clf.fit(X_train, y_train)
    print(bag_clf.oob_score_)
    y_pred = bag_clf.predict(X_test)
    print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred))
    # 当基决策器有predict_proba()时）决策函数会对每个训练实例返回类别概率
    print(bag_clf.oob_decision_function_)


if __name__ == "__main__":
    # 生成月亮数据集
    X, y = make_moons(n_samples=1000, noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    hard_and_sort_vote(X_train, X_test, y_train, y_test)

    bag_and_paste(X_train, X_test, y_train, y_test)