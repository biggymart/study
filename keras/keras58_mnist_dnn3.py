# 다차원 데이터의 Dense 모델
# keras58_mnist_dnn2.py 카피
# 쌩 Dense모델로 다차원 데이터를 돌리기 (x, y 4차원)

#1. data
import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1-1. preprocessing
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.

# x와 y의 shape을 4차원으로 맞추기 위해서 임의로 x를 y에 넣어보자
y_train = x_train ### Point1 ###
y_test = x_test
print(y_train.shape)
print(y_test.shape)
# (60000, 28, 28, 1)
# (10000, 28, 28, 1)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, input_shape=(28,28,1), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1)) # linear 상태로 나옴, y 출력값이랑 같아짐 ### Point 1 ###

# model.summary()

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1) 

model.compile(loss='mse', optimizer='adam', metrics=['acc']) ### Point 2 ###
model.fit(x_train, y_train, epochs=50, batch_size=256, callbacks=[es, rlr], validation_split=0.5)

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[mse, acc] :", loss) #[mse, acc] : [0.2505415380001068, 0.5]

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape) # (10000, 28, 28, 1)

