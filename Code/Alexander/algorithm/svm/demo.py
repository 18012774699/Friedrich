from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

rng = np.random
# svr = joblib.load('svr.pkl')        # 读取模型

x = rng.uniform(1, 100, (100, 1))
y = 5 * x + np.sin(x) * 5000 + 2 + np.square(x) + rng.rand(100, 1) * 5000

# 自动选择合适的参数
svr = GridSearchCV(SVR(),
                   param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
svr.fit(x, y.ravel())
# joblib.dump(svr, 'svr.pkl')        # 保存模型

xneed = np.linspace(0, 100, 100)[:, None]
y_pre = svr.predict(xneed)  # 对结果进行可视化：
plt.scatter(x, y, c='k', label='data', zorder=1)
# plt.hold(True)
plt.plot(xneed, y_pre, c='r', label='SVR_fit')
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()
plt.show()
print(svr.best_params_)
