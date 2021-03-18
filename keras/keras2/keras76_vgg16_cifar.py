# 실습: cifar10에 vgg16 전이학습 적용할 것

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout # Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)
# plt.imshow(x_train[0], 'gray')
# plt.show()

#1-1. 데이터 전처리
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 32, 32, 3)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. compile and fit
es = EarlyStopping(monitor='val_loss',
            patience=10,
            mode='auto',
            verbose=1)
re = ReduceLROnPlateau(monitor='val_loss',
            factor=0.2,
            patience=5,
            verbose=1)

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc'])
hist = model.fit(x_train, y_train,
            epochs=100,
            batch_size=32,
            callbacks=[es],
            validation_split=0.2)

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test)
idx = 10
for i in range(idx):
    print(np.argmax(y_test[i]), np.argmax(y_pred[i]), end='/')

# Visualization
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

# 전이학습 미적용 결과
# [categorical_crossentropy, acc] : [1.1093847751617432, 0.6141999959945679]
# 3 3/8 8/8 8/0 8/6 4/6 6/1 1/6 6/3 3/1 1/

# 전이학습 적용 결과
# [categorical_crossentropy, acc] : [1.769873857498169, 0.6007000207901001]
# 3 3/8 1/8 8/0 1/6 6/6 6/1 3/6 4/3 6/1 5/