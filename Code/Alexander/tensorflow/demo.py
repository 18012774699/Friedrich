import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    init = tf.compat.v1.global_variables_initializer()  # prepare an init node
    f = x*x*y + y + 2
# ===========================

# way1
sess = tf.compat.v1.Session(graph=graph)
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)

print(result)
sess.close()
# ===========================

# way2
with tf.compat.v1.Session(graph=graph) as sess2:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)
# ===========================

# way3
with tf.compat.v1.Session(graph=graph) as sess3:
    init.run()  # actually initialize all the variables
    result = f.eval()
    print(x.graph is tf.compat.v1.get_default_graph())
    print(result)
# ===========================

