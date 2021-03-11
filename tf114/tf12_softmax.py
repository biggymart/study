import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1], # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0], # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0], # 0
          [1, 0, 0],]
x_pred = [[1, 11, 7, 9]]


x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float', [None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([1, 3]), name='bias')
# 각 행에 대해서 값을 하나씩 더해주기 위함

# ** 중요 **
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# (None, 4) * (4, 3) + (1, 3)
# 최종 shape (N, 3); 결과값을 감싸는 activation은 softmax

# ** 중요 **
# categorical_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # test the damn hypothesis (training)
    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost],
            feed_dict={x:x_data, y:y_data})
        
        # verbose
        if step % 200 == 0:
            print(step, cost_val)

    # 훈련이 끝나서 hypothesis에 w, b 업데이트 됨
    # predict
    a = sess.run(hypothesis, feed_dict={x:x_pred})
    print(a, sess.run(tf.argmax(a, 1)))
    # [[0.80384046 0.19088006 0.00527951]] all adds up to 1
    # argmax(a, 1) a 중에서 가장 큰 값에 1을 줘라