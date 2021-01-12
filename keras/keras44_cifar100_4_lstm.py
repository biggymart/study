# keras43_cifar10_4_lstm.py 카피

#1. data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape) # (60000, 28, 28)

#1-1. 데이터 전처리
x_train = x_train.reshape(60000, 7 * 7, 16).astype('float32')/255. ### Point1 ###
x_test = x_test.reshape(10000, 7 * 7, 16)/255. 
# (x_test.reshape(x_test.shape[0], idx[1], idx[2])) # idx[1] * idx[2] ==  x_test.shape[1] * x_test.shape[2]

from tensorflow.keras.utils import to_categorical # OneHotEncoding from tensorflow
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.activations import relu, softmax

model = Sequential()
model.add(LSTM(32, input_shape=(7 * 7, 16), activation=relu)) ### Point2 ###
model.add(Dense(256, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(10, activation=softmax))

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