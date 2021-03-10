# shape의 중요성
import tensorflow as tf
tf.set_random_seed(66)

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]] # (5,3)
y_data = [[152],
          [185],
          [180],
          [205],
          [142]] # (5,1)


x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal(shape=[3,1]), name='weight')
# x * w 는 (5,3) * (3,1) 이니까 output의 shape은 (5,1) 
# x * w 의 output shape == y의 shape; 
# hypothesis와 실재값 비교해야 하기 때문
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

# [실습] 만들어보시오
cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) # 0.00001
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_data, y:y_data})
        if step % 2000 == 0:
            print("epoch :", step, "\ncost:", cost_val, "\nhypothesis :\n", hy_val)


