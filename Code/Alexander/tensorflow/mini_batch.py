import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime

housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{0}行,{1}列".format(m, n))
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

#
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = r"./tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

graph = tf.Graph()
with graph.as_default():
    X = tf.compat.v1.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.compat.v1.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    training_op = optimizer.minimize(mse)

    init = tf.compat.v1.global_variables_initializer()
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())

n_epochs = 10
batch_size = 100
# 计算总批次数
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices]  # not shown
    y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
    return X_batch, y_batch


with tf.compat.v1.Session(graph=graph) as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                simple_value = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                summary_str = tf.compat.v1.Summary()
                summary_str.value.add(tag="mse_summary", simple_value=simple_value)
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
    file_writer.close()
    print(best_theta)
