# 다중분류
# OneHotEncoding: 이진분류와 구분되는 특징 (x와 y의 차원 모두 2차원, 'y를 벡터화' != 'y에 대한 전치리')
# Output Layer node == number of classification

import numpy as np
from sklearn.datasets import load_iris

#1. data
# x, y = load_iris(return_X_y=True) # 교육용에서만 지원해줌
dataset = load_iris()
x = dataset.data
y = dataset.target

'''
print(dataset.DESCR)
print(dataset.feature_names, "\n", dataset.target_names)
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] ['setosa' 'versicolor' 'virginica']
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print(x[:5])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]
print(y) 
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# OneHotEncoding, tensorflow
from tensorflow.keras.utils import to_categorical # tensorflow 2.0의 방식
# from keras.utils.np_utils import to_categorical # 옛날 keras 방식, 앞에 tensorflow 붙이면 안 됨
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train) # 0 -> [1. 0. 0.]
# print(y_train.shape) # (120, 3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(3, activation='softmax'))
# softmax takes a vector, sigmoid takes a scalar, 
# ** 분류하고자 하는 숫자만큼 노드를 잡아라 **, 노드값의 합은 1 (e.g. 0.49, 0.31, 0.2 => 0, 파이그래프로 그리기도 함)
# 다중분류는 y도 전처리해줘야 함; 1차원 scalar인 데이터를 2차원 vertor인 데이터로 만들어줌
# 분류에서는 0, 1, 2 의 의미가 '2가 1의 2배이다'가 아니다 그래서...
# 값의 위치만 지정, 원핫인코딩 https://wikidocs.net/22647 (sklearn, keras 두 버전 있음)
# to_cateorical은 반드시 0에서부터 시작한다, 만약에 0이 없으면 0을 넣어줘야 함
# one_hot_coding은 그렇게 할 필요가 없음

# softmax 함수는 이전 노드들에서 input이 들어온 값들을 함수에 통과시켜서 각 인덱스 값에 대해 0 ~ 1 사이의 값으로 분배해줌 (총합이 1), 그 중 가장 큰 값으로 그 노드가 선택됨

#3. compile and fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 다중분류에서 loss는 반드시 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')

model.fit(x_train, y_train, epochs=10000, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print(loss)

# 과제1> 프레딕트값을 원하는 값 (이진분류: 0,1 및 다중분류)으로 변경해서 출력되도록 코드를 변경하시오, argmax 함수 이용

y_pred = model.predict(x_test[-5:])
for i in y_pred:
    print("(인덱스) 꽃이름 :", np.argmax(i), dataset.target_names[np.argmax(i)], ", 값 :", np.max(i))
print(y_test[-5:])

# 결과 keras22_1_iris1_keras
# [0.12124720960855484, 0.9666666388511658]
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.9984915
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.9997532
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.94546694
# (인덱스) 꽃이름 : 1 versicolor , 값 : 0.5085082
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.94557166
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]