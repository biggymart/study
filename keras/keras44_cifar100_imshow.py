import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # 
print(x_test.shape, y_test.shape)   # 

print(x_train[0])
print(y_train[0]) #

print(x_train[0].shape) #
print(y_train.min(), y_train.max()) #

plt.imshow(x_train[0], 'gray')
plt.show()
