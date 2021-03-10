# 가상환경 없이 tf 버전2에서 버전1을 사용하는 방법

import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello World")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)
# 세션을 통과하지 않으면 자료형의 구조만 나옴

sess = tf.Session()
print(sess.run(hello))
# b'Hello World'

# 3가지 자료형: Constant, Variable, Placeholder

