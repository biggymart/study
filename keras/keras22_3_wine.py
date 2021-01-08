# 와인의 품종을 확인하는 데이터셋
# 실습> DNN 모델을 완성시켜라

'''
print(dataset.DESCR)
print(dataset.feature_names) # 13개
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

print(x)
[[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]
 [1.320e+01 1.780e+00 2.140e+00 ... 1.050e+00 3.400e+00 1.050e+03]
 [1.316e+01 2.360e+00 2.670e+00 ... 1.030e+00 3.170e+00 1.185e+03]
 ...
 [1.327e+01 4.280e+00 2.260e+00 ... 5.900e-01 1.560e+00 8.350e+02]
 [1.317e+01 2.590e+00 2.370e+00 ... 6.000e-01 1.620e+00 8.400e+02]
 [1.413e+01 4.100e+00 2.740e+00 ... 6.100e-01 1.600e+00 5.600e+02]]
print(y)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
print(x.shape)(178, 13)
print(y.shape)(178,)
'''
#1. data
import numpy as np
from sklearn.datasets import load_wine
dataset = load_wine() # sklearn은 클래스의 인스턴스 만드는 식
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
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(13,)))
model.add(Dense(3, activation='softmax')) # print(y) 에서 3가지로 분류된 것 확인함

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류에서 loss는 반드시 categorical_crossentropy
model.fit(x_train, y_train, epochs=10000, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test[-5:])
for i in y_pred:
    print("(인덱스) 와인이름 :", np.argmax(i), dataset.target_names[np.argmax(i)], ", 값 :", np.max(i))
print(y_test[-5:])

# 결과 keras22_3_wine.py
# Epoch 1730/10000
# [categorical_crossentropy, acc] : [0.11018853634595871, 0.9722222089767456]
# (인덱스) 와인이름 : 0 class_0 , 값 : 1.0
# (인덱스) 와인이름 : 2 class_2 , 값 : 0.99999976
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.9999982
# (인덱스) 와인이름 : 1 class_1 , 값 : 0.9998067
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.99287087
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]]