import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.] # type(x) == <class 'list'>
y = [2., 4., 6.]

w = tf.placeholder(tf.float32) # Tensor("Placeholder:0", dtype=float32)

hypothesis = x * w # Tensor("mul:0", dtype=float32)
cost = tf.reduce_mean(tf.square(hypothesis - y)) # Tensor("Mean:0", shape=(), dtype=float32)

w_history = []
cost_history = []

with tf.Session() as sess:
    for i in range(-30, 50):
        curr_w = i * 0.1
        curr_cost = sess.run(cost, feed_dict={w:curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

print("===================")
print(w_history)
print("===================")
print(cost_history)
print("===================")

plt.plot(w_history, cost_history)
plt.show()