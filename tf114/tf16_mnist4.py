# layer에 initializer 사용
# 한번에 6만개 훈련시키지 말고 
# batch 100 단위로 짤라서 훈련시켜

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

### 2. 모델 구성
# w1 = tf.Variable(tf.random_normal([28*28, 100]), name='weight1')
w1 = tf.get_variable('weight1', shape=[28*28, 100],
                     initializer=tf.contrib.layers.xavier_initializer()) # kernel_initializer
b1 = tf.Variable(tf.random_normal([100]), name='bias1')
layer1 = tf.nn.elu(tf.matmul(x, w1) + b1) # elu 활성화 함수; relu, selu도 가능
layer1 = tf.nn.dropout(layer1, keep_prob=0.8) # 30퍼센트를 킵한다

w2 = tf.get_variable('weight2', shape=[100, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([128]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

w3 = tf.get_variable('weight3', shape=[128, 64],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([64]), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

w4 = tf.get_variable('weight4', shape=[64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10]), name='bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)


#3. 컴파일, 훈련(다중분류)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005).minimize(cost)

training_epochs = 75
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

# Acc : 0.8444