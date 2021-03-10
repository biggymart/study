# 회귀
from sklearn.datasets import load_diabetes
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 10]) # feature 10개
y = tf.placeholder(tf.float32, shape=[None, 1])

# [실습]

