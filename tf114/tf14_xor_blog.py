# https://blog.daum.net/ejleep1/921
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_xor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)

# hyperparameter
learning_rate = 0.00001
training_epochs = 2000
display_steps = 1000

n_input = 2
dof1 = 1

x = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.float32, shape=[None, dof1])

W1 = tf.Variable(tf.random_normal([n_input, dof1]))
b1 = tf.Variable(tf.random_normal([dof1]))
W2 = tf.Variable(tf.random_normal([n_input, dof1]))
b2 = tf.Variable(tf.random_normal([dof1]))

# hypothesis = fn(x, W1, b1, W2, b2)
hypothesis = (tf.matmul(x, W2) + b2) * (tf.matmul(x, W1) + b1)

# cost/loss function
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1)), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        _, c, w1, B1, w2, B2 = sess.run([optimizer, cost, W1, b1, W2, b2], feed_dict={x:x_xor, y:y_xor})

        if (epoch+1) % display_steps == 0:
            print("Epoch :", (epoch+1), "Cost :", c, w1, B1, w2, B2)
    print("Optimization Finished!")

    # Accuracy report
    h, p, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x:x_xor, y:y_xor})
    print("\nHypothesis :\n", h, "\nCorrect :\n", p, "\nAccuracy :", a)
    print(p.shape)
    print(p)
    sess.close()