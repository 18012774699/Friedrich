from sklearn.datasets import fetch_openml
# from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def display_digit_jpg(X, y):
    print(X.shape)
    print(y.shape)

    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

    print(y[36000])


# 准确率/召回率曲线
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, rate: float = 0.9):
    # 计算满足条件的thresholds
    precision_threshold = thresholds[np.where(abs(precisions - rate) < 0.0001)][0]
    recall_threshold = thresholds[np.where(abs(recalls - rate) < 0.0001)][0]

    plt.plot(thresholds, precisions[:-1], "b-", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.hlines(rate, thresholds.min(), thresholds.max(), color="red", linestyles="dashed", label="90%")  # 横线
    plt.vlines(precision_threshold, 0, 1, color="blue", linestyles="dashed",
               label=str(precision_threshold) + "=" + str(rate * 100) + "%")  # 竖线
    plt.vlines(recall_threshold, 0, 1, color="green", linestyles="dashed",
               label=str(recall_threshold) + "=" + str(rate * 100) + "%")  # 竖线
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()
    return int(precision_threshold), int(recall_threshold)


# 绘制 ROC 曲线
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.show()


# 对性能的评估
def performance_evaluation():
    pass


if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', data_home="../datasets")
    # mnist = fetch_mldata('MNIST original', data_home="../datasets")
    print(mnist)

    X, y = mnist["data"], mnist["target"]
    # display_digit_jpg(X, y)

    # 切分测试集
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # 打乱训练集
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    # 训练一个二分类器
    y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)

    if 0:
        # 对性能的评估
        # 混淆矩阵
        # 混淆矩阵中的每一行表示一个实际的类, 而每一列表示一个预测的类。
        # N: 预测非5，P: 预测是5；TF是否正确
        '''
         TN | FP
        ----|----
         FN | TP
        '''
        # 默认返回预测值
        y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
        confusion_matrix(y_train_5, y_train_pred)
        print(confusion_matrix(y_train_5, y_train_pred))
        # print(confusion_matrix(y_train_5, y_train_5))

        print(precision_score(y_train_5, y_train_pred))  # TP/(TP+FP)
        print(recall_score(y_train_5, y_train_pred))  # TP/(TP+FN)
        print(f1_score(y_train_5, y_train_pred))

        # 根据决策分数，决策
        y_scores = sgd_clf.decision_function([X[36000]])
        print(y_scores)
        threshold = 0
        print(y_scores > threshold)

        threshold = 200000
        print(y_scores > threshold)

        # 指定返回决策分数
        y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

        # 使用Matplotlib画出准确率/召回率曲线，根据曲线设定阈值
        precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
        precision_threshold, recall_threshold = plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

        y_train_pred_90 = (y_scores > precision_threshold)
        print(precision_score(y_train_5, y_train_pred_90))
        print(recall_score(y_train_5, y_train_pred_90))

        # ROC 曲线
        fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
        plot_roc_curve(fpr, tpr)
        plt.show()

        # 一个比较分类器优劣的方法是：测量ROC曲线下的面积（AUC，area under the curve）
        print(roc_auc_score(y_train_5, y_scores))

        # 训练一个RandomForestClassifier
        forest_clf = RandomForestClassifier(random_state=42)
        y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
        y_scores_forest = y_probas_forest[:, 1]  # 正反概率

        # ROC 曲线
        fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

        plt.plot(fpr, tpr, "b:", label="SGD")
        plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
        plt.legend(loc="lower right")
        plt.show()
        print(roc_auc_score(y_train_5, y_scores_forest))
        print(precision_score(y_train_5, y_scores_forest > 0.5))
        print(recall_score(y_train_5, y_scores_forest > 0.5))
    # =========================

    # 误差分析
    sgd_clf.fit(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()

    # 用混淆矩阵，看误差数据
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums

    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()

    # =========================

