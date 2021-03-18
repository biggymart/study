# 레이어에서 정규화
# kernel_initializer (가중치 초기화): He(relu계열), Xavier(sigmoid, tanh계열)
# bias_initializer  
# kernel_regularizer

# BatchNormalization
# Dropout (둘을 동시에 안 쓴다; GAN에선 같이 쓴다)

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2 # l norm


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1-1. 데이터 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255. 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), padding='same',
                 strides=1, input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (2,2), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (2,2), kernel_regularizer=l1(l1=0.01)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))



#3. compile and fit
es = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss='categorical_crossentropy',
     optimizer='Adam',
     metrics=['acc']
) 
hist = model.fit(x_train,
     y_train,
     epochs=50,
     batch_size=1024,
     callbacks=[es],
     validation_split=0.2
)

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test)
idx = 10
for i in range(idx):
    print(np.argmax(y_test[i]), np.argmax(y_pred[i]), end='/')


# #5. Visualization
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6)) # 도화지 면적을 잡아줌, 가로가 10, 세로가 6

# plt.subplot(2, 1, 1) # 2행 1열 짜리 그래프를 만들겠다, 그 중 첫번째
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()

# plt.title('Cost Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# plt.subplot(2, 1, 2) # 2행 1열 그래프 중 두번째
# plt.plot(hist.history['acc'], marker='.', c='red', label='accuracy')
# plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_accuracy')
# plt.grid() # 격자

# plt.title('Accuracy')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right') # keras36_hist0.py line 69처럼, label 값을 지정할수도 있다

# plt.show()