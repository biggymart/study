from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score

#1. OR data
x_data = [[0, 0], [1,0], [0,1], [1, 1]]
y_data = [0, 1, 1, 0] ### Takeaway1 ###

#2. model
model = LinearSVC() ### old model ###

#3. fit
model.fit(x_data, y_data)

#4. score and predict
y_pred = model.predict(x_data)
print(x_data, "'s predicted result :", y_pred)

result = model.score(x_data, y_data)
print("model.score :", result)

acc = accuracy_score(y_data, y_pred)
print("accuracy_score :", acc)

# [[0, 0], [1, 0], [0, 1], [1, 1]] 's predicted result : [1 1 1 1]
# model.score : 0.5
# accuracy_score : 0.5

# [[0, 0], [1, 0], [0, 1], [1, 1]] 's predicted result : [0 1 1 1]
# model.score : 0.75
# accuracy_score : 0.75
# this is the best it can do...