# 문제 푸는 순서 국룰 
# load, slice, shape -> make the bloody model run
# parameter tuning

import tensorflow as tf
from sklearn.datasets import load_boston

dataset = load_boston()
x_data = dataset.data
y_data = dataset.target

print(x_data.shape) # (506, 13)
print(y_data.shape) # (506,)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])


from sklearn.metrics import r2_score