# keras23_LSTM1.py 카피
#1. data
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape) # (4, 3)
print(y.shape) # (4,)

x = x.reshape(4, 3, 1) 

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

model = Sequential()
model.add(SimpleRNN(10, activation='relu', input_shape=(3,1))) 
model.add(Dense(20))
model.add(Dense(10)) 
model.add(Dense(1))

model.summary()

#3. compile and fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

x_pred = np.array([5, 6, 7]) # (3,)
x_pred = x_pred.reshape(1, 3, 1)

#4. evaluate and predict
loss = model.evaluate(x, y)
print(loss)

result = model.predict(x_pred)
print(result)

# 결과 SimpleRNN
# 2.2976364562055096e-05
# [[8.020421]]

# 결과 LSTM1
# 0.01851184293627739
# [[8.077794]]

# 일반적으로 LSTM의 성능이 SimpleRNN보다 더 좋다고 한다
# 이유는 연산이 더 많기 때문이다