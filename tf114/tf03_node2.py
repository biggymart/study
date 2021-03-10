# [실습]
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈

import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

add = tf.add(node1, node2)
sub = tf.subtract(node1, node2)
mul = tf.multiply(node1, node2)
div = tf.divide(node1, node2)

sess = tf.Session()
print('add :', sess.run(add))
print('sub :', sess.run(sub))
print('mul :', sess.run(mul))
print('div :', sess.run(div))

# add : 5.0
# sub : -1.0
# mul : 6.0
# div : 0.6666667

# 설명은 없고 예시만 몇 개만 있음
# https://daeson.tistory.com/250
# 자세한 설명과 함께 예시도 다양하게 제공
# https://excelsior-cjh.tistory.com/151

