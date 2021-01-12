# 과제> np.load를 활용하여 모델을 완성하시오

#1. data
import numpy as np
x_data = np.load('../data/npy/cancer_x.npy')
y_data = np.load('../data/npy/cancer_y.npy')

# keras46_MC_6_cancer.py 카피
#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
for i in range(10):
    model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ### Point1 ###
modelpath = '../data/modelCheckpoint/k46_cancer_{epoch:02d}-{val_loss:.4f}.hdf5' ### Point2 ###
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') ### Point3 ###
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=10000, validation_split=0.2, callbacks=[early_stopping, check_point]) ### Point4 ###

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("binary_crossentropy, acc :", loss)

y_predict = model.predict(x_test[-10:])
y_pred = list(map(int,np.round(y_predict,0)))
y_predict = np.transpose(y_predict)
y_pred = np.transpose(y_pred)
print(y_pred)
print(y_test[-10:])

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

plt.title('Acc')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# 결과
# binary_crossentropy, acc : [0.1141447052359581, 0.9649122953414917]
# [0 1 1 1 0 1 0 1 1 1]
# [0 1 1 1 0 1 0 1 1 1]