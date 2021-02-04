# 다차원 데이터의 Dense 모델
# keras57_Reduce_lr.py 카피
# 4차원 데이터인 이미지도 DNN모델로 구성 가능함

#1. data
import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1-1. preprocessing
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.activations import relu, softmax

model = Sequential()
model.add(Dense(64, input_shape=(28,28,1), activation=relu)) ### Point 1 ###
model.add(Dropout(0.5))
model.add(Conv2D(filters=32, kernel_size=(2,2)))
model.add(Flatten()) # 결과 나오기 전에 Flatten 해줘야 Dense로 출력됨
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
# [categorical_crossentropy, acc] : [0.21947629749774933, 0.9715999960899353]