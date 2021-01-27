from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score

#1. OR data
x_data = [[0, 0], [1,0], [0,1], [1, 1]]
y_data = [0, 1, 1, 0] 

#2. model
# model = LinearSVC()
model = SVC() ### Takeaway1 ### new model

#3. fit
model.fit(x_data, y_data)

#4. score and predict
y_pred = model.predict(x_data)
print(x_data, "'s predicted result :", y_pred)

result = model.score(x_data, y_data)
print("model.score :", result)

acc = accuracy_score(y_data, y_pred)
print("accuracy_score :", acc)

# [[0, 0], [1, 0], [0, 1], [1, 1]] 's predicted result : [0 1 1 0]
# model.score : 1.0
# accuracy_score : 1.0
# 인공지능의 겨울이 해결되었다! 종이를 대각선으로 접는 것과 비슷함 (원리 알고 싶으면 개인적으로 찾아보길)