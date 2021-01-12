# 시계열이야 아니야?
# (어거지로 맞추면) 시계열임, 연속된 데이터가 될 수 있는 거잖아
# 성능을 보고나서 판단해보자! 먼저 이게 뭐라고 미리 판단해보지 말고

# DNN (N, 764), 
# LSTM (N, 764, 1) (1개씩 잘라서 쓰므로 느림, 꼭 경험해봐라)
#   혹은 (N, 28*14, 2) 혹은 (N, 28*7, 4) 혹은 (N, 7*7, 16)
#   LSTM(input_shape=(28*28, 1)) 이렇게 될거야
# CNN (N, 28, 28) 

# 이미지 데이터는 선형회귀 및 시계열 데이터로 해석할 수 있어
# 구글 데이터 코랩 이용하도록 하자

#1. data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
# [categorical_crossentropy, acc] : [0.4545525908470154, 0.838699996471405]
# 7 7/2 2/1 1/0 0/4 4/1 1/4 4/9 9/5 5/9 9/