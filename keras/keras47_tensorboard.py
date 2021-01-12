# keras45_ModelCheckPoint_mnist.py 카피
# (1) cmd > pip list > tensorboard 2.4.0 확인
# (2) 웹 서버 구동하듯이 cmd 입력 (cf. dir /w)
# graph>tensorboard --logdir=. #graph 폴더를 지칭함
# (3) 웹 브라우저를 키고 127.0.0.1:6006 주소로 이동 (로컬 주소, 코드 번호 6006)
# (4) 혼자서 실습할 때 그래프 폴더를 비우고 해야 다른 결과물과 안 꼬임

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard ### Point1 ###
modelpath = './modelCheckPoint/k47_tensorboard_{epoch:02d}-{val_loss:.4f}.hdf5' ### Point2 ###
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='loss', patience=3, mode='auto')
tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True) ### Point3 ###

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=30, batch_size=1024, callbacks=[es, cp, tb], validation_split=0.2) ### Point4 ###

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test)
idx = 10
for i in range(idx):
    print(np.argmax(y_test[i]), np.argmax(y_pred[i]), end='/')

# Visualization
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

# 과제: 텐서보드 파라미터 알아보기