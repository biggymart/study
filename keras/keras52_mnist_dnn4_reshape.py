# 다차원 데이터의 Dense 모델
# keras58_mnist_dnn3.py 카피
# Reshape layer

#1. data
import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1-1. preprocessing
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.

y_train = x_train
y_test = x_test
print(y_train.shape)
print(y_test.shape)
# (60000, 28, 28, 1)
# (10000, 28, 28, 1)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Reshape ### Point 1 ###
from tensorflow.keras.activations import relu
model = Sequential()
model.add(Dense(64, input_shape=(28,28,1), activation=relu))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(28 * 28, activation=relu)) ### Point 2 ### # 노드의 갯수를 reshape과 맞춰야 함
model.add(Reshape((28, 28, 1))) # 왜 이렇게 괄호가 많냐? 다른 인자 디폴트로 들어가는 게 있기 때문
model.add(Dense(1)) # linear 상태로 나옴, y 출력값이랑 같아짐

# model.summary()

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1) 

from tensorflow.keras.optimizers import Adam
model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=50, batch_size=256, callbacks=[es, rlr], validation_split=0.5) 

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[mse, acc] :", loss) # [mse, acc] : [0.007130193989723921, 0.8142049908638]

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape) # (10000, 28, 28, 1)

