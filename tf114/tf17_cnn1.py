import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN이니까 4차원
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

#2. 모델구성

# CNN
w1 = tf.get_variable("w1", shape=[3, 3, 1, 32]) # kernel size w, kernel size h, channel, filters(or output)
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME') # strides=[1,2,2,1]
print(L1)
# Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)

# Conv2D(filter, kernel_size, input_shape)
# Conv2D(32, (3,3), input_shape=(28, 28, 1)) # 가로, 세로, 채널
# # of param: (kernel_size + 1) * channel * filter; (2*2+1) * 1 * 10
# 이 레이어를 통과한 후 shape: 28 by 28 by 32

L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L1) # ksize 가 중요함
# Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

w2 = tf.get_variable("w2", shape=[3, 3, 32, 64])
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

w3 = tf.get_variable("w3", shape=[3, 3, 64, 128])
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)
# Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)

w4 = tf.get_variable("w4", shape=[3, 3, 128, 64])
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)
# Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

# Flatten
L_flat = tf.reshape(L4, [-1, 2*2*64])
print(L_flat)
# Tensor("Reshape:0", shape=(?, 256), dtype=float32)

# Dense
w5 = tf.get_variable("w5", shape=[2*2*64, 50],
    initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([50]), name='b5')
L5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=0.8)
print(L5)
# Tensor("dropout/mul_1:0", shape=(?, 50), dtype=float32)

w6 = tf.get_variable("w6", shape=[50, 32],
    initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([32]), name='b6')
L6 = tf.nn.selu(tf.matmul(L5, w6) + b6)
L6 = tf.nn.dropout(L6, keep_prob=0.8)
print(L6)

w7 = tf.get_variable("w7", shape=[32, 10],
    initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([10]), name='b7')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7)
print(hypothesis)

#3. 컴파일, 훈련
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

training_epochs = 10
batch_size = 100
total_batch = int(len(x_train)/batch_size)
# 주의: 전체 데이터(i.e. 60000)가 batch_size(i.e. 100)로 
# 나누어 떨어지지 않으면 그만큼 데이터 손실 발생

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0

        for i in range(total_batch): # 1 epoch에 600번 돈다
            start = i * batch_size   # 0    100 200 ...
            end = start + batch_size # 100  200 300 ...

            batch_x, batch_y = x_train[start:end], y_train[start:end]
            feed_dict = {x:batch_x, y:batch_y}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict) # 한 배치의 cost
            avg_cost += c/total_batch
        
        print('Epoch :', '%04d' %(epoch + 1),
              'cost = {:.9f}'.format(avg_cost))

    print('훈련 끝')

    prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# Acc : 0.9871