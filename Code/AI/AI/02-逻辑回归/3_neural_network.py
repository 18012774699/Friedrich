# 神经网络案例
from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.]]
y = [0, 1]

# MLP（Multi-Layer Perceptron），即多层感知器
# alpha：                L2正则项惩罚的权重
# activation：           激活函数
# hidden_layer_sizes：   隐藏层大小，每个数代表每层的节点数，5 x 2（5x2个参数）
# tol：                  threshold，阈值
clf = MLPClassifier(solver='sgd', alpha=1e-5, activation='logistic', hidden_layer_sizes=(5, 2), max_iter=2000, tol=1e-4)
clf.fit(X, y)

# 预测两个输入的结果
predicted_value = clf.predict([[2., 2.], [-1., -2.]])
print(predicted_value)
predicted_proba = clf.predict_proba([[2., 2.], [-1., -2.]])
print(predicted_proba)

# 打印神经网络的层次结构、所有层的参数
print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])
