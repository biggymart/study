from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

# 데이터 구성
dataset = load_iris()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

enc = OneHotEncoder()
y_data = enc.fit_transform(y_data.reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 텐서머신
x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float', [None, 3])
w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([1, 3]), name='bias')

# 훈련지표
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # (training)
    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost],
            feed_dict={x:x_train, y:y_train})
        
        # verbose
        if step % 200 == 0:
            print(step, cost_val)

    # predict
    a = sess.run(hypothesis, feed_dict={x:x_test})
    print('Raw Prediction :\n', a, '\nPrediction :\n', sess.run(tf.argmax(a, 1)), '\nActual :\n', sess.run(tf.argmax(y_test,1)))
    print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))
    print('Acc :', accuracy_score(y_test, sess.run(predicted, feed_dict={x:x_test})))