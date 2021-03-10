# 실습
# x와 y를 placeholder로

import tensorflow as tf
tf.set_random_seed(66)

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b],
                 feed_dict={x_train: [1,2,3], y_train: [3,5,7]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
