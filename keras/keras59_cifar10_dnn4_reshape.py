# 다차원 댄스 모델
# x -> y : (n, 32, 32, 3) -> (n, 32, 32, 3)
# Reshape layer 사용

#1. data
import numpy as np
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#1-1. 데이터 전처리
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255. 
x_test = x_test.reshape(10000, 32, 32, 3)/255. 

y_train = x_train
y_test = x_test
print(y_train.shape)
print(y_test.shape)
# (50000, 32, 32, 3)
# (10000, 32, 32, 3)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Reshape
from tensorflow.keras.activations import relu
model = Sequential()
model.add(Dense(64, input_shape=(32,32,3), activation=relu))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32 * 32, activation=relu))
model.add(Reshape((32, 32, 1)))
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1) 

from tensorflow.keras.optimizers import Adam
model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=50, batch_size=256, callbacks=[es, rlr], validation_split=0.5) 

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[mse, acc] :", loss) # [mse, acc] : [0.05848466232419014, 0.002527994103729725]

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape) # (10000, 32, 32, 1)
