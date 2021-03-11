# https://blog.naver.com/PostView.nhn?blogId=laonple&logNo=221377293443


# 요구사항: DNN 모델로 구성하시오
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

# hyperparameters
training_epochs = 2000
display_steps = 1000

# load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape) # (60000, 28 * 28)

# DNN 위해서 2차원으로 축소
x_train = x_train.reshape((60000, 28*28))
x_test = x_test.reshape((10000, 28*28))

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = enc.transform(y_test.reshape(-1,1)).toarray()
# print(y_train.shape) # (60000, 10)

# scaling
x_train, x_test = x_train/255.0, x_test/255.0

# 텐서머신
x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, 10])

### 모델 구성
# hidden 1
W1 = tf.Variable(tf.random_normal([28*28, 100]), name='weight1')
b1 = tf.Variable(tf.random_normal([100]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# hidden 2
W2 = tf.Variable(tf.random_normal([100, 50]), name='weight2')
b2 = tf.Variable(tf.random_normal([50]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

# output layer
W3 = tf.Variable(tf.random_normal([50, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

# cost/loss function
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
###

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost],
             feed_dict={x:x_train, y:y_train})

        if (epoch+1) % display_steps == 0:
            print("Epoch :", (epoch+1), "Cost :", c)
    print("Optimization Finished!")

    # Accuracy report
    h, p, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x:x_test, y:y_test})
    print("\nHypothesis :\n", h, "\nCorrect :\n", p, "\nAccuracy :", a)
    print(p.shape)
    print(p)
    sess.close()

