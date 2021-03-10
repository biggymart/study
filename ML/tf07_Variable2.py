import tensorflow as tf

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

# [실습]
#1. sess.run()
#2. InteractiveSession()
#3. Var.eval(session=sess)
# hypothesis를 출력하는 코드를 만드시오

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(hypothesis)
print("aaa :", aaa)
sess.close()

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = hypothesis.eval(session=sess)
print("bbb :", bbb)
sess.close()

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval()
print("ccc :", ccc)
sess.close()