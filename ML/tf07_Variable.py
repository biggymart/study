import tensorflow as tf
tf.compat.v1.set_random_seed(66)

W = tf.Varaiable(tf.compat.v1.random_normal([1]), name='weight')
print(W)

# Session
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(W)
print("aaa :", aaa)
sess.close()

# Session + eval
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = W.eval(session=sess)
print("bbb :", bbb)
sess.close()

# InteractiveSession + eval
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval()
print("ccc :", ccc)
sess.close()



