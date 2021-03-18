import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten


x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 실습: 모델을 만드시오

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150, 150, 1), padding='same'))
model.add(MaxPooling2D(padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32)

# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# print("acc :", acc[-1])
# print("val_acc :", val_acc[:-1])

loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss, acc: ', loss, acc)
# loss, acc:  5.766897678375244 0.5