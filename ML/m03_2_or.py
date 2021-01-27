from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score

#1. OR data
x_data = [[0, 0], [1,0], [0,1], [1, 1]]
y_data = [0, 1, 1, 1] ### Takeaway1 ###

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

# [[0, 0], [1, 0], [0, 1], [1, 1]] 's predicted result : [0 1 1 1]
# model.score : 1.0
# accuracy_score : 1.0
# we can confirm that the model got the prediction with 100% accuracy