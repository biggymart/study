# learning_rate 수정해서
# epoch를 2000보다 작게 만드시오

import tensorflow as tf
tf.set_random_seed(66)

# 모델 구성
x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b # linear
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # MSE
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # 훈련
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b],
                 feed_dict={x_train: [1,2,3], y_train: [3,5,7]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
        
    # 예측
    pred1 = sess.run(hypothesis,
        feed_dict={x_train: [4]})
    pred2 = sess.run(hypothesis,
        feed_dict={x_train: [5,6]})
    pred3 = sess.run(hypothesis,
        feed_dict={x_train: [6,7,8]})  
    print(pred1)
    print(pred2)      
    print(pred3)      

