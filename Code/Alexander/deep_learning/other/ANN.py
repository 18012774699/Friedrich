import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

iris = load_iris()
X = iris.data[:, (2, 3)]  # 花瓣长度，宽度
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])
print(y_pred)
# =======================================

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
# X_train = tf.convert_to_tensor(X_train, dtype=tf.float32) / 255.
# y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
# y_train = tf.one_hot(y_train, depth=10)


def train_input_fn(features, labels, batch_size, training=True):
    # 将输入数据转为数据集（dataset）
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # 打乱数据集顺序、重复执行，再批处理数据
    if training:
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


feature_columns = tf.infer_real_valued_columns_from_input(X_train)
# 两个隐藏层的 DNN（一个具有 300 个神经元，另一个具有 100 个神经元）和一个具有 10 个神经元的 softmax 输出层进行分类
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_columns)
dnn_clf.train(input_fn=lambda: train_input_fn(X_train, y_train, 50), steps=40000)
y_pred = list(dnn_clf.predict(X_test))
print(accuracy_score(y_test, y_pred))
print(dnn_clf.evaluate(X_test, y_test))
