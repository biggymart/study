# 분류
# 이진분류 (binary classification): 결과값이 2개인 경우
# 예시> 여러 특성(키, 몸무게, 나이 등)으로 성별을 예측하는 경우
# 다중분류(categorical classification): 결과값이 여러개인 경우
# 예시> 여러 특성으로 국적을 예측하는 경우
# 3 things to remember: (1) layer > activation='sigmoid, (2) compile > loss='binary_crossentropy', and (3) metrics=['acc']

# 유방암 예측모델

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data
datasets = load_breast_cancer()
'''
print(datasets.DESCR) # (569, 30)
print(datasets.feature_names, "\n", datasets.target_names)
'''
x = datasets.data
y = datasets.target
'''
print(x.shape) # (569, 30)
print(y.shape) # (569,)
print(x[:5])
print(y[:100]) # 0, 1 로만 이루어져서 분류라고 알 수 있음
'''
# 전처리 (MinMaxScaler, train_test_split) 알아서 하기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
for i in range(10):
    model.add(Dense(10, activation='relu'))
# activation: 넘겨줄 값인 output 에 제한을 걸어주는 것
# hidden layer 없을 수도 있음 (ML), activation='relu' 는 치역의 범위가 0 ~ 무한
model.add(Dense(1, activation='sigmoid')) 
# output layer, 없으면 default 는 linear, 치역범위: -무한 ~ 무한; activation='sigmoid' 는 치역의 범위가 0 ~ 1로 한정됨
# *** 이진분류면 output layer에 activation='sigmoid' ***
# https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

#3. compile and fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# *** 당분간 "이진분류면 이 loss='binary_crossentropy'를 쓴다"라고만 외워 *** (나중에 한꺼번에 loss 정리 예정)
# metrics=['acc'] 도 어지간하면 이렇게 설정

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 

model.fit(x_train, y_train, epochs=10000, validation_split=0.2, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("binary_crossentropy, acc :", loss)

y_predict = model.predict(x_test[-10:])
# 0과 1 "사이"의 값이 나오고, 
# compile의 binary_crossentropy에서 0인지 1인지 판단하게 됨, 그로부터 accuracy 측정함
# 만약 0, 1 로 출력되길 원한다면 if문 사용 (e.g> 0.5 이상이면 1, 아니면 0)
y_pred = list(map(int,np.round(y_predict,0)))
y_predict = np.transpose(y_predict)
y_pred = np.transpose(y_pred)
print(y_pred)
print(y_test[-10:])

# 실습1> acc 를 0.98 이상으로 끌어올릴 것
# 실습2> 일부 데이터에서 원래 y 값과 y_predict 값과 비교해볼 것

# 결과
# binary_crossentropy, acc : [0.11954214423894882, 0.9561403393745422]
# [0 1 1 1 0 1 0 1 1 1]
# [0 1 1 1 0 1 0 1 1 1]