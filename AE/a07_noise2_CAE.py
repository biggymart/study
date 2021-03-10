# CAE 방식으로 노이즈 만들고 복원해주기
# shape이 문제다! 항상 shape 조심

# 시간 남으면 male female에 AE 적용해보기

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Conv2DTranspose
import matplotlib.pyplot as plt
import random

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1)/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) # 0 ~ 0.1 사이의 노이즈 정규분포 만들기
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # 0~1 사이로 고정시키다
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

### 핵 심 내 용 #############################################################
COLOR_CHANNEL = 1
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(128, (2,2), 1, padding='same', input_shape=(28, 28, COLOR_CHANNEL)))
    model.add(Conv2D(32, (2,2), 1, padding='same', activation='relu'))
    model.add(Dense(units=hidden_layer_size, activation='relu'))
    model.add(Conv2DTranspose(32, (2,2), 1, padding='same', activation='relu'))
    model.add(Conv2DTranspose(128, (2,2), 1, padding='same', activation='relu'))
    model.add(Dense(COLOR_CHANNEL, 'sigmoid')) # 흑백은 1, 컬러는 3
    return model
#############################################################################

model = autoencoder(hidden_layer_size=154) # 95%로 복원되는 수치 = 154 (즉, 복원수준)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train_noised, x_train, epochs=10) # 노이즈 있는 놈과 없는 놈 비교

output = model.predict(x_test_noised) # 노이즈가 제거되었는지 확인

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
        (ax11, ax12, ax13, ax14, ax15)) =\
            plt.subplots(3, 5, figsize=(20, 7))
        
# 이미지 5개 무작위 선정
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지를 중간에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISED", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음 제거한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()