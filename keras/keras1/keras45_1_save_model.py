# keras45_ModelCheckPoint_mnist.py 카피

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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.activations import relu, softmax

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), padding='valid', strides=1, input_shape=(28,28,1), activation=relu))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(10, activation=softmax))

model.save('../data/h5/k51_1_model1.h5') # 이 모델을 저장, Point1

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k51_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=50, batch_size=1024, callbacks=[early_stopping, check_point], validation_split=0.2)

model.save('../data/h5/k51_1_model2.h5') # 이 모델을 저장, Point2

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

# 결과
# [categorical_crossentropy, acc] : [0.0728115364909172, 0.9764000177383423]
# 7 7/2 2/1 1/0 0/4 4/1 1/4 4/9 9/5 6/9 9/