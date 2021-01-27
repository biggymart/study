# 인공지능의 역사 중 두 차례의 겨울 있는데
# 1번째 겨울: and, or, nor, nand gate 다 되는데 xor이 안 된다
# https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5

import numpy as np
from sklearn.svm import LinearSVC # support vector classifier
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