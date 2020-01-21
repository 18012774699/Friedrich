# 整流线性单元（ReLU）
import tensorflow as tf

n_features = 3

graph = tf.Graph()
with graph.as_default():
    def relu(X):
        threshold = tf.compat.v1.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.compat.v1.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="max")

    X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = []
    for relu_index in range(5):
        with tf.compat.v1.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
            relus.append(relu(X))
    output = tf.add_n(relus, name="output")

with tf.compat.v1.Session(graph=graph) as sess:
    pass
    # sess.run(output)

