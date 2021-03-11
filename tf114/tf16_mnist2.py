import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

### 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 전처리 (정규화 원핫인코딩)
x_train = x_train.reshape(60000,28*28).astype('float32')/255
x_test = x_test.reshape(10000,28*28).astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 텐서머신 정의
x = tf.placeholder('float', [None, 28*28])
y = tf.placeholder('float', [None, 10])

w = tf.Variable(tf.random_normal([28*28, 10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')

### 2. 모델 구성
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(2001):
        _, c = sess.run([optimizer, cost], feed_dict={x:x_train, y:y_train})

        if epoch % 200 == 0:
            print("Epoch :", (epoch), "Cost :", c)
    print("Optimization Finished!")

    # Accuracy report
    h, p, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x:x_test, y:y_test})
    print("\nHypothesis :\n", h, "\nCorrect :\n", p, "\nAccuracy :", a)
    print(p.shape)
    print(p)
    sess.close()




