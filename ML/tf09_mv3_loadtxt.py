import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

dataset = np.loadtxt('../data/csv/data-01-test-score.csv', delimiter=',')
x_data = dataset[:, :-1]
# print(x_data.shape) # (25,3)
y_data = dataset[:, -1].reshape(-1,1)
print(y_data.shape) # (25,)

x_test = [[73, 80, 75],
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]]
y_test = [[152],
          [185],
          [180],
          [196],
          [142]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal(shape=[3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

# [실습] 만들어보시오
cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) # 0.00001
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 훈련
    for step in range(2001): # epoch
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_data, y:y_data})

        # verbose
        if step % 2000 == 0:
            print("epoch :", step, "\ncost:", cost_val, "\nhypothesis :\n", hy_val)
    print("============================")
    # 예측
    pred = sess.run(hypothesis,
        feed_dict={x: x_test})
    print("예측값\tvs\t실제값\t차이")
    for i in range(len(pred)):
        diff = y_test[i] - pred[i]
        print(pred[i], "\t", y_test[i], "\t", diff)

