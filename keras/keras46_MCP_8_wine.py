# keras38_dropout5_wine.py 카피

#1. data
import numpy as np
from sklearn.datasets import load_wine
dataset = load_wine()
x = dataset.data
y = dataset.target

#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from tensorflow.keras.utils import to_categorical # OneHotEncoding from tensorflow
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(13,)))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax')) # print(y) 에서 3가지로 분류된 것 확인함

#3. compile and fit

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ### Point1 ###
modelpath = '../data/modelCheckPoint/k46_wine_{epoch:02d}-{val_loss:.4f}.hdf5' ### Point2 ###
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') ### Point3 ###
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류에서 loss는 반드시 categorical_crossentropy
hist = model.fit(x_train, y_train, epochs=10000, validation_split=0.2, callbacks=[early_stopping, check_point]) ### Point4 ###

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test[-5:])
for i in y_pred:
    print("(인덱스) 와인이름 :", np.argmax(i), dataset.target_names[np.argmax(i)], ", 값 :", np.max(i))
print(y_test[-5:])

# Visualization ### Point5 ###
import matplotlib.pyplot as plt
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
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid() # 격자

plt.title('acc')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# 결과
# [categorical_crossentropy, acc] : [0.7237894535064697, 0.8888888955116272]
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.6287289
# (인덱스) 와인이름 : 2 class_2 , 값 : 0.5475988
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.5257476
# (인덱스) 와인이름 : 2 class_2 , 값 : 0.4276691
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.4068869
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]]