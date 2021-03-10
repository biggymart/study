# y = wx + b

import tensorflow as tf
tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias') 
# 정규분포에 의한 랜덤한 값 하나만 집어넣어라

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(W), sess.run(b))
# [0.06524777] [1.4264158]

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) # 손실의 최소화

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001): # epoch=2000
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))



