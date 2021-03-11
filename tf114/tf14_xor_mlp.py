import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_xor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)

# hyperparameter
training_epochs = 2000
display_steps = 1000

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])
# W * x + b


### multi-layer 구성
# hidden 1 (노드 10개)
W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, W1) + b1)
# output (None, 10)

# hidden 2 (노드 7개)
W2 = tf.Variable(tf.random_normal([10, 7]), name='weight2')
b2 = tf.Variable(tf.random_normal([7]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
# output (None, 7)

# output layer
W3 = tf.Variable(tf.random_normal([7, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)
# output (None, 1)


# model.add(Dense(10, input_dim=2, activation='sigmoid'))
# model.add(Dense(7, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))


# cost/loss function
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1)), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x:x_xor, y:y_xor})

        if (epoch+1) % display_steps == 0:
            print("Epoch :", (epoch+1), "Cost :", c)
    print("Optimization Finished!")

    # Accuracy report
    h, p, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x:x_xor, y:y_xor})
    print("\nHypothesis :\n", h, "\nCorrect :\n", p, "\nAccuracy :", a)
    print(p.shape)
    print(p)
    sess.close()