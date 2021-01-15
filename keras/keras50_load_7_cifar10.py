# 과제> np.load를 활용하여 모델을 완성하시오

#1. data
import numpy as np
x_train = np.load('../data/npy/cifar10_x_train.npy')
y_train = np.load('../data/npy/cifar10_y_train.npy')
x_test = np.load('../data/npy/cifar10_x_test.npy')
y_test = np.load('../data/npy/cifar10_y_test.npy')

# keras46_MC_2_cifar10.py 카피
#1-1. preprocessing
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255. # 0 ~ 1 사이의 숫자로 만듦
x_test = x_test.reshape(10000, 32, 32, 3)/255. # 0 ~ 1 사이의 숫자로 만듦
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
                strides=1, input_shape=(32,32,3), activation=relu))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(10, activation=softmax))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k46_cifar10_{epoch:02d}-{val_loss:.4f}.hdf5' 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=100, batch_size=1024, callbacks=[early_stopping, check_point], validation_split=0.2)

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test)
idx = 10
for i in range(idx):
    print(np.argmax(y_test[i]), np.argmax(y_pred[i]), end='/')

# Visualization 
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6)) 

plt.subplot(2, 1, 1) 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_accuracy')
plt.grid()

plt.title('Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# 결과
# [categorical_crossentropy, acc] : [1.0685149431228638, 0.6259999871253967]
# 3 3/8 8/8 8/0 8/6 4/6 6/1 1/6 6/3 3/1 1/