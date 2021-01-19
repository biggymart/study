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

# x와 y의 shape을 4차원으로 맞추기 위해서 임의로 x를 y에 넣어보자
y_train = x_train ### Point1 ###
y_test = x_test
print(y_train.shape)
print(y_test.shape)
# (50000, 32, 32, 3)
# (10000, 32, 32, 3)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, input_shape=(32,32,3), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1) 

model.compile(loss='mse', optimizer='adam', metrics=['acc']) 
model.fit(x_train, y_train, epochs=50, batch_size=256, callbacks=[es, rlr], validation_split=0.5)

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[mse, acc] :", loss)

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape) # (10000, 32, 32, 1)