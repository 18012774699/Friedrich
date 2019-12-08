# 逻辑回归
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from time import time

# 加载鸢尾花数据集
iris = datasets.load_iris()
print(list(iris.keys()))  # 打印字典关键字
# print(iris['DESCR'])        # 数据集描述（包括相关度）
print(iris['feature_names'])

# X[:,  m:n]，即取所有数据的第m到n-1列数据，含左不含右
X = iris['data'][:, 3:] # petal width (cm)
# X = iris['data']        # 所有维度
print(X)

y = iris['target']  # 分类号
print(y)
# y = (iris['target'] == 2).astype(np.int)
print(y)

# Utility function to report best scores
# def report(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")
#
#
# start = time()

# 用于交叉验证的超参数
param_grid = {"tol": [1e-4, 1e-3, 1e-2], "C": [0.4, 0.6, 0.8]}

# multi_class='ovr'，二分类; 'multinomial'=多分类
log_reg = LogisticRegression(multi_class='ovr', solver='sag')
log_reg.fit(X, y)

# 交叉验证，评估模型最使用的超参数
grid_search = GridSearchCV(log_reg, param_grid=param_grid, cv=3)
grid_search.fit(X, y)

# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)

# linspace：返回[0,3]间100等分数据，reshape：以为数据转成矩阵
X_new = np.linspace(0, 3, 100).reshape(-1, 1)
print(X_new)

y_proba = log_reg.predict_proba(X_new)  # 概率
y_hat = log_reg.predict(X_new)          # 分类号
print(y_proba)
print(y_hat)

print("W1 =", log_reg.coef_)
print("W0 =", log_reg.intercept_)

print("W1 =", grid_search.best_estimator_)

plt.plot(X_new, y_proba[:, 2], 'g-', label='Iris-Virginica')
plt.plot(X_new, y_proba[:, 1], 'r-', label='Iris-Versicolour')
plt.plot(X_new, y_proba[:, 0], 'b--', label='Iris-Setosa')
plt.show()

print(log_reg.predict([[1.7], [1.5]]))
