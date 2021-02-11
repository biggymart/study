# https://taeguu.tistory.com/24
# https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342
# https://hackmd.io/@bouteille/S1WvJyqmI
# https://ivo-lee.tistory.com/91
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt
import os
import random

# 파일을 numpy 혹은 pd 형식으로 불러온 다음에 100 이하의 값은 0으로 replace해서 돋보이게 해보자 (plt로 확인)

TRAIN_DIR = 'C:/data/data_2/dirty_mnist_2nd/'
print(os.listdir(TRAIN_DIR)[:5])

sample = plt.imread(os.path.join(TRAIN_DIR,random.choice(os.listdir(TRAIN_DIR))))
plt.show(sample)

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
# model.add(MaxPool2D(strides=2))
# model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
# model.add(MaxPool2D(strides=2))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(84, activation='relu'))
# model.add(Dense(10, activation='softmax'))