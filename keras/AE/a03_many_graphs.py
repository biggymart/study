import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import random
import sys

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784)/255.

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

### 동적 변수 생성 ###################################
mod = sys.modules[__name__]
lst = ['01', '02', '04', '08', '16', '32']
for i in lst:
    # 방법 1: 
    # setattr(obj, attr_name, value) is (more or less) same as mod.attr_name = value 
    setattr(mod, 'model_{}'.format(i), autoencoder(hidden_layer_size=int(i)))

    # 방법 2:
    # globals()['model_{}'.format(i)] = autoencoder(hidden_layer_size=int(i))
### https://data-newbie.tistory.com/353
### https://muzukphysics.tistory.com/225
### https://stackoverflow.com/questions/4743913/dynamically-create-call-variables-in-python3
### 여기까진 ok ###

### 퀘스트: 생성된 동적 변수를 for 문에 호출하여 동적으로 compile, fit, predict를 하여라
    # getattr(mod, 'model_{}'.format(i))
    # print("node {}개 시작".format(i))
    # mod.compile(optimizer='adam', loss='binary_crossentropy')
    # mod.fit(x_train, x_train, epochs=10)


# ################################
# for i in lst:
#     print('node {}개 시작'.format(i))
#     'model_{}'.format(i).compile(optimizer='adam', loss='binary_crossentropy')
#     'model_{}'.format(i).fit(x_train, x_train, epochs=10)



# for idx in range(5) :
#    print(getattr(mod,  'object_{}'.format(idx)))
# https://data-newbie.tistory.com/353
#################################

print("node 1개 시작")
model_01.compile(optimizer='adam', loss='binary_crossentropy')
model_01.fit(x_train, x_train, epochs=10)

print("node 2개 시작")
model_02.compile(optimizer='adam', loss='binary_crossentropy')
model_02.fit(x_train, x_train, epochs=10)

print("node 4개 시작")
model_04.compile(optimizer='adam', loss='binary_crossentropy')
model_04.fit(x_train, x_train, epochs=10)

print("node 8개 시작")
model_08.compile(optimizer='adam', loss='binary_crossentropy')
model_08.fit(x_train, x_train, epochs=10)

print("node 16개 시작")
model_16.compile(optimizer='adam', loss='binary_crossentropy')
model_16.fit(x_train, x_train, epochs=10)

print("node 32개 시작")
model_32.compile(optimizer='adam', loss='binary_crossentropy')
model_32.fit(x_train, x_train, epochs=10)

output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_04 = model_04.predict(x_test)
output_08 = model_08.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)

fig, axes = plt.subplots(7, 5, figsize=(15, 15))

random_imgs = random.sample(range(output_01.shape[0]), 5)
outputs = [x_test, output_01, output_02, output_04, output_08, output_16, output_32]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28),
                cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()


# PCA 는 95퍼 유사도가 174 였음
# 그러니까 여기도 95퍼 유사도 나오려면 그 정도 노드 있어야함



