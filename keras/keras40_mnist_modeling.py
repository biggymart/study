# 인공지능계의 Hello, World라 불리는 mnist

# DNN 모델로 구성 가능
# CNN으로 들어갈 수 있게 x.shape = (60000, 28, 28) --> x.reshape(-1, x.shape[1], x.shape[2], 1)
# DNN으로 들어갈 수 있게 x.shape = (60000, 28, 28) --> x.reshape(-1, x.shape[1]*x.shape[2])
# x.train.shape = (N, 28, 28) = (N, 28*28) = (N, 764)
# 과제> Dense 모델로 구성, input_shape=(28*28,)

# LSTM 모델로 구성 가능
# 시계열이야 아니야?
# (어거지로 맞추면) 시계열임, 연속된 데이터가 될 수 있는 거잖아
# 성능을 보고나서 판단해보자! 먼저 이게 뭐라고 미리 판단해보지 말고

# 정리
# DNN (N, 764), 
# LSTM (N, 764, 1) (1개씩 잘라서 쓰므로 느림, 꼭 경험해봐라) // 혹은 (N, 28*14, 2) 혹은 (N, 28*7, 4) 혹은 (N, 7*7, 16) // LSTM(input_shape=(28*28, 1)) 이렇게 될거야
# CNN (N, 28, 28) 

#1. data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1-1. 데이터 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. # 0 ~ 1 사이의 숫자로 만듦
x_test = x_test.reshape(10000, 28, 28, 1)/255. # 0 ~ 1 사이의 숫자로 만듦
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
                strides=1, input_shape=(28,28,1), activation=relu))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
model.add(Flatten())
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

# 실습> 코드를 완성하시오
# 지표는 acc, 0.985 이상
# [categorical_crossentropy, acc] : [0.04836886376142502, 0.9836000204086304]

# 응용
# y_test 10개와 y_pred 10개를 출력하시오
# 출력 예시
# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?)

# 7 7/2 2/1 1/0 0/4 4/1 1/4 4/9 9/5 5/9 9/