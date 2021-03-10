import tensorflow as tf

sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')
# 변수는 sess.run을 통과시키기 전에 초기화를 시켜줘야 함

# 초기화
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

print(sess.run(x))
# [2.]