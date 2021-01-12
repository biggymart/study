# keras40_mnist2_cnn.py 카피
# ModelCheckPoint는 5개의 사항(Point) 변경해주면 됨

#1. data
import numpy as np
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ### Point1 ###
modelpath = './modelCheckPoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' ### Point2 ###
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') ### Point3 ###
# filepath: 최저점을 찍을 때마다 해당 지점의 가중치가 들어간 파일을 만듦, 궁극의 w가 저장된다
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), 
              metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=50, batch_size=1024, 
                callbacks=[early_stopping, check_point], validation_split=0.2) ### Point4 ###

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test)
idx = 10
for i in range(idx):
    print(np.argmax(y_test[i]), np.argmax(y_pred[i]), end='/')

# Visualization ### Point5 ###
import matplotlib.pyplot as plt # Matplotlib 한글 미지원
plt.figure(figsize=(10, 6)) # 도화지 면적을 잡아줌, 가로가 10, 세로가 6

plt.subplot(2, 1, 1) # 2행 1열 짜리 그래프를 만들겠다, 그 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) # 2행 1열 그래프 중 두번째
plt.plot(hist.history['acc'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_accuracy')
plt.grid() # 격자

plt.title('Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# 과제
# val_loss와 loss가 교차하는 시점이 생김 (과적합 진입), 
# val_loss의 최저점을 매번 최신화하여 찍어주는 ModelCheckPoint를 적용해라.