# keras51_1_save_model.py 카피
# Study Objective: Know the difference of two models saved at two different points (each before fit and after fit)

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

model.save('../data/h5/k52_1_model1.h5') ### model1.h5 file ONLY saves the model ###

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' ### this hdf5 file saves the minimum val_loss ###
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='loss', patience=5, mode='auto')

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=50, batch_size=1024, callbacks=[es, cp], validation_split=0.2)

model.save('../data/h5/k52_1_model2.h5')         ### model2.h5 file saves BOTH the model AND the weight ###
model.save_weights('../data/h5/k52_1_weight.h5') ### weight.h5 file saves ONLY the weight ###

#4. evaluate
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

# Result of keras52_1_save_weight.py
# 313/313 [==============================] - 0s 1ms/step - loss: 0.0764 - acc: 0.9762
# [categorical_crossentropy, acc] : [0.07644817233085632, 0.9761999845504761]

# After running this file, you will have three h5 files and one hdf5 file.
# Each of these files remembers different information according to the comments stated above.
# To outline:
# k52_1_model1.h5:          model ONLY
# k52_1_mnist_00-0000.hdf5: model AND min val_loss
# k52_1_model2.h5:          model AND weight
# k52_1_weight.h5:          weight ONLY