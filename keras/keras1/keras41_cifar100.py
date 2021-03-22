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

# cifar100

#1-1. 데이터 전처리
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255. # 0 ~ 1 사이의 숫자로 만듦
x_test = x_test.reshape(10000, 32, 32, 3)/255. # 0 ~ 1 사이의 숫자로 만듦
# (x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

from tensorflow.keras.utils import to_categorical # OneHotEncoding from tensorflow
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.activations import relu, softmax

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), padding='valid',
                strides=1, input_shape=(32,32,3), activation=relu))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(100, activation=softmax))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc']) # 다중분류에서 loss는 반드시 categorical_crossentropy
model.fit(x_train, y_train, epochs=100, batch_size=1024, callbacks=[early_stopping], validation_split=0.2)

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test)
idx = 10
for i in range(idx):
    print(np.argmax(y_test[i]), np.argmax(y_pred[i]), end='/')

# 결과
# [categorical_crossentropy, acc] : [2.8704073429107666, 0.3034999966621399]
# 49 12/33 80/72 15/51 51/71 71/92 31/15 80/14 7/23 71/0 83/