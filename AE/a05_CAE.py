# Convolutional --> CNN으로 구성
# 개념
# Conv <-> ConvTranspose
# pooling <-> upsampling


import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, LeakyReLU
import matplotlib.pyplot as plt


(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32')/255 # 28*28 = 784
x_test = x_test/255.


def autoencoder():
    inputs = Input(shape=(28,28,1))
    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_1 = x

    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_2 = x

    x = Conv2DTranspose(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_2) # 잔차학습 (Residual learning)
    x = Dropout(0.4)(x)
    x = LeakyReLU()(x)
    x = x

    x = Conv2DTranspose(filters=1,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_1) # 흑백이면 필터 1개, 컬러면 필터 3개
    x = Dropout(0.4)(x)
    x = LeakyReLU()(x)
    outputs = x
    model = Model(inputs = inputs,outputs=outputs)
    return model

model = autoencoder()
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) =\
    plt.subplots(2, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()





