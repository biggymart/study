# 다차원 데이터의 Dense 모델
# keras43_cifar10_2_cnn.py 카피
# x -> y : (n, 32, 32, 3) -> (n, 32, 32, 3)

#1. data
import numpy as np
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#1-1. preprocessing
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 32, 32, 3)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.activations import relu, softmax

model = Sequential()
model.add(Dense(64, input_shape=(32,32,3), activation=relu))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32, kernel_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation=relu))
model.add(Dense(10, activation=softmax))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train, y_train, epochs=50, batch_size=256, callbacks=[es, rlr], validation_split=0.5)

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)
# [categorical_crossentropy, acc] : [1.9604480266571045, 0.508899986743927]