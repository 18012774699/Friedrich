# 评估指标代码调用
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# 下载并导入数据集
mnist = fetch_mldata('MNIST original', data_home='./test_data_home')
print(mnist)
print("------------------")

X, y = mnist['data'], mnist['target']
print(X.shape, y.shape)     # 打印行列数
print("------------------")

some_digit = X[36000]   # 第36000张图片数据
print(some_digit)
some_digit_image = some_digit.reshape(28, 28)
print(some_digit_image)

# 画出图片
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
#            interpolation='nearest')
# plt.axis('off')
# plt.show()
print("------------------")

# [:60000] = 0~59999
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[:60000]
# 先生成随机列表，然后打乱顺序
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
print(y_test_5)
# 逻辑回归分类器，判断是否=5
sgd_clf = SGDClassifier(loss='log', random_state=42, max_iter=1000, tol=1e-4)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))

# 3折交叉验证
# skfolds = StratifiedKFold(n_splits=3, random_state=42)
# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = y_train_5[train_index]
#     X_test_folds = X_train[test_index]
#     y_test_folds = y_train_5[test_index]
#
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_folds)
#     print(y_pred)
#     n_correct = sum(y_pred == y_test_folds)
#     print('正确率：%.2f%%' % (100 * n_correct / len(y_pred)))
# 3折交叉验证，等价于上面
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy'))
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='precision'))
print("------------------")


class Never5Classifier(BaseEstimator):  # 继承基类分类器
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        # 都返回0，shape(len(X),1)
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
# 本身只有10% = 5
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy'))

# 混淆矩阵
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(confusion_matrix(y_train_5, y_train_pred))
y_train_perfect_prediction = y_train_5
print(confusion_matrix(y_train_5, y_train_perfect_prediction))

# 精确率、召回率、f1_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
print(sum(y_train_pred))
print(f1_score(y_train_5, y_train_pred))

# 决策边界和信心值
sgd_clf.fit(X_train, y_train_5)
# 对预测结果的有符号信心值，Z = w^TX
y_score = sgd_clf.decision_function([some_digit])
print(y_score)

# 设置信心阈值
threshold = 0
y_some_digit_pred = (y_score > threshold)
print(y_some_digit_pred)

threshold = -15000
y_some_digit_pred = (y_score > threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
#print(y_scores)
#precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
#print(precisions, recalls, thresholds)

"""
# 绘制precision_recall_curve
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'r--', label='Recall')
    plt.xlabel("Threshold")
    plt.legend(loc='upper left')
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
"""

y_train_pred_90 = (y_scores > 70000)
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# 绘制ROC曲线
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')


plot_roc_curve(fpr, tpr)
plt.show()

print(roc_auc_score(y_train_5, y_scores))

# 绘制随机森林ROC曲线
"""
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, 'b:', label='SGD')
plt.plot(fpr_forest, tpr_forest, label='Random Forest')
plt.legend(loc='lower right')
plt.show()

print(roc_auc_score(y_train_5, y_scores_forest))
"""