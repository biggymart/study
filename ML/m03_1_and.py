# 인공지능의 역사중에 겨울이 두 번 온다
# https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5

# 1번째 겨울: xor의 문제
# 머신러닝의 여러 기법이 있는데, 회귀도 있는데, 분류도 있음
# 문제 발생: 
# and, or, nor, nand gate 다 되는데 xor이 안 된다

from sklearn.svm import LinearSVC # support vector classifier
import numpy as np
from sklearn.metrics import accuracy_score # classification metrics in sklearn ### Takeaway1 ###

#1. AND data
x_data = [[0, 0], [1,0], [0,1], [1, 1]]
y_data = [0, 0, 0, 1]

#2. model
model = LinearSVC()

#3. fit
model.fit(x_data, y_data)

#4. score and predict
y_pred = model.predict(x_data)
print(x_data, "'s predicted result :", y_pred)

result = model.score(x_data, y_data)
print("model.score :", result)

acc = accuracy_score(y_data, y_pred)
print("accuracy_score :", acc)

# [[0, 0], [1, 0], [0, 1], [1, 1]] 's predicted result : [0 0 0 1]
# model.score : 1.0
# accuracy_score : 1.0
# we can confirm that the model got the prediction with 100% accuracy