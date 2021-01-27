# DL w/o hidden layer
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

#1. OR data
x_data = [[0, 0], [1,0], [0,1], [1, 1]]
y_data = [0, 1, 1, 0] 

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. compile and fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. score and predict
y_pred = model.predict(x_data)
print(x_data, "'s predicted result :", y_pred)

result = model.evaluate(x_data, y_data) # score 대신 evaluate
print("model.score :", result[1]) # acc를 보려고 함

# [[0, 0], [1, 0], [0, 1], [1, 1]] 's predicted result : [[0.5339807 ] [0.3409582 ] [0.2921327 ] [0.15706792]]
# model.score : 0.25
