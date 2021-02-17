# GPU 분산처리 (distribute)
# https://keras.io/guides/distributed_training/ 공식문서 참고

# CPU는 n_jobs로 가능

#1. data
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1-1. 데이터 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255. 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5)


##############################
strategy = tf.distribute.MirroredStrategy(cross_device_ops=\
    tf.distribute.HierarchicalCopyAllReduce()
)

with strategy.scope():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same',
                    strides=1, input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (2,2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #3. compile and fit
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=['acc'])
##############################

hist = model.fit(x_train, y_train, epochs=50, batch_size=32,
                callbacks=[es], validation_split=0.2)

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test)
idx = 10
for i in range(idx):
    print(np.argmax(y_test[i]), np.argmax(y_pred[i]), end='/')