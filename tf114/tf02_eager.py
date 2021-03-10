# 즉시 실행 모드 (sess.run 없이 실행함)
# from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

print(tf.executing_eagerly()) # False

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) # False

print(tf.__version__)

hello = tf.constant("Hello World")
print(hello)

# sess = tf.Session() # 1.13까지
sess = tf.compat.v1.Session()
print(sess.run(hello))

# AttributeError: module 'tensorflow' has no attribute 'Session'
# 2버전부터 Session을 아예 삭제함

# b'Hello World'
